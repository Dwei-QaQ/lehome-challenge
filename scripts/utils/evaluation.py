import os
import argparse
import gymnasium as gym
import torch
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from scripts.eval_policy import PolicyRegistry
from scripts.eval_policy.base_policy import BasePolicy

from scripts.utils.eval_utils import (
    convert_ee_pose_to_joints,
    save_videos_from_observations,
    calculate_and_print_metrics,
)

from lehome.utils.record import (
    RateLimiter,
    get_next_experiment_path_with_gap,
    append_episode_initial_pose,
)
from scripts.utils.dataset_record import create_teleop_interface
from lehome.devices.action_process import convert_sim_state_to_so101_leader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .common import stabilize_garment_after_reset
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers for real-robot torque control and state synchronisation
# ---------------------------------------------------------------------------



def _enable_real_torque(teleop) -> None:
    """Enable torque on the real robot motor buses."""
    try:
        if hasattr(teleop, 'left_so101_leader') and hasattr(teleop, 'right_so101_leader'):
            for arm in (teleop.left_so101_leader, teleop.right_so101_leader):
                bus = getattr(arm, '_bus', None)
                if bus is not None:
                    bus.enable_torque()
        elif hasattr(teleop, '_bus') and teleop._bus is not None:
            teleop._bus.enable_torque()
    except Exception as e:
        logger.warning(f"Failed to enable real robot torque: {e}")


def _disable_real_torque(teleop) -> None:
    """Disable torque on the real robot motor buses (make arm passive)."""
    try:
        if hasattr(teleop, 'left_so101_leader') and hasattr(teleop, 'right_so101_leader'):
            for arm in (teleop.left_so101_leader, teleop.right_so101_leader):
                bus = getattr(arm, '_bus', None)
                if bus is not None:
                    bus.disable_torque()
        elif hasattr(teleop, '_bus') and teleop._bus is not None:
            teleop._bus.disable_torque()
    except Exception as e:
        logger.warning(f"Failed to disable real robot torque: {e}")


def _write_state_to_real(teleop, state_np: np.ndarray) -> None:
    """Write simulation joint state (radians) to real robot motors.

    Uses convert_sim_state_to_so101_leader (inverse of convert_action_from_so101_leader)
    to convert radians → motor normalized values before writing.
    """
    try:
        if hasattr(teleop, 'left_so101_leader') and hasattr(teleop, 'right_so101_leader'):
            left_arm = teleop.left_so101_leader
            right_arm = teleop.right_so101_leader
            left_bus = getattr(left_arm, '_bus', None)
            right_bus = getattr(right_arm, '_bus', None)
            if (
                left_bus is not None
                and right_bus is not None
                and hasattr(left_bus, 'write_desired_position')
                and hasattr(right_bus, 'write_desired_position')
                and len(state_np) >= 12
            ):
                left_motor = convert_sim_state_to_so101_leader(state_np[:6], left_arm.motor_limits)
                right_motor = convert_sim_state_to_so101_leader(state_np[6:12], right_arm.motor_limits)
                left_bus.write_desired_position(left_motor)
                right_bus.write_desired_position(right_motor)
        elif hasattr(teleop, '_bus') and teleop._bus is not None:
            if hasattr(teleop._bus, 'write_desired_position'):
                motor_vals = convert_sim_state_to_so101_leader(state_np[:6], teleop.motor_limits)
                teleop._bus.write_desired_position(motor_vals)
    except Exception as e:
        logger.debug(f"Failed to write state to real robot: {e}")


def run_evaluation_loop(
    env: DirectRLEnv,
    policy: BasePolicy,
    args: argparse.Namespace,
    ee_solver: Optional[Any] = None,
    is_bimanual: bool = False,
    garment_name: Optional[str] = None,
    policy_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Core evaluation loop.
    Refactored to be agnostic of specific model implementations.
    """

    # --- Dataset Recording Setup (Optional) ---
    eval_dataset = None
    record_dataset = None
    json_path_eval = None
    json_path_record = None
    episode_index_eval = 0
    episode_index_record = 0

    if args.save_datasets:
        is_bi_arm = "Bi" in args.task or "bi" in args.task.lower()
        action_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ]
        if is_bi_arm:
            joint_names = [f"left_{n}" for n in action_names] + [f"right_{n}" for n in action_names]
        else:
            joint_names = action_names
        dim = len(joint_names)

        features: Dict[str, Dict[str, Any]] = {
            "observation.state": {"dtype": "float32", "shape": (dim,), "names": joint_names},
            "action":            {"dtype": "float32", "shape": (dim,), "names": joint_names},
        }
        image_keys = ["top_rgb", "left_rgb", "right_rgb"] if is_bi_arm else ["top_rgb", "wrist_rgb"]
        for key in image_keys:
            features[f"observation.images.{key}"] = {
                "dtype": "video", "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
            }

        def _make_dataset(root_path: Path, repo_id: str) -> LeRobotDataset:
            return LeRobotDataset.create(
                repo_id=repo_id,
                fps=args.step_hz,
                root=get_next_experiment_path_with_gap(root_path),
                use_videos=True,
                image_writer_threads=8,
                image_writer_processes=0,
                features=features,
            )

        eval_dataset   = _make_dataset(Path(args.eval_dataset_path), "lehome_eval")
        record_dataset = _make_dataset(
            Path(getattr(args, "record_dataset_path", "Datasets/record")), "lehome_record"
        )
        json_path_eval   = eval_dataset.root   / "meta" / "garment_info.json"
        json_path_record = record_dataset.root / "meta" / "garment_info.json"
        logger.info(f"eval dataset   → {eval_dataset.root}")
        logger.info(f"record dataset → {record_dataset.root}")

    all_episode_metrics = []
    logger.info(f"Starting evaluation: {args.num_episodes} episodes")
    rate_limiter = RateLimiter(args.step_hz)

    # optional teleop interface (override policy)
    teleop = None
    if getattr(args, "teleop_device", None):
        try:
            teleop = create_teleop_interface(env, args)
            if teleop:
                teleop.manual_control = False
                for arm_attr in ('left_so101_leader', 'right_so101_leader'):
                    arm = getattr(teleop, arm_attr, None)
                    if arm is not None:
                        arm.other_key_enable = True
                if hasattr(teleop, 'other_key_enable'):
                    teleop.other_key_enable = True
            logger.info(f"Teleop device '{args.teleop_device}' initialized for evaluation.")
        except Exception as e:
            logger.warning(f"Failed to initialize teleop device: {e}")

    sync_to_real = getattr(args, "sync_to_real", False) and teleop is not None
    if sync_to_real:
        _enable_real_torque(teleop)
        logger.info("sync_to_real enabled: real robot will follow simulation state at every step.")

    # ---------------------------------------------------------------
    # Runtime keyboard hints
    # ---------------------------------------------------------------
    _sep = "=" * 56
    logger.info(_sep)
    logger.info("KEYBOARD CONTROLS")
    logger.info(_sep)
    if getattr(args, "enable_pause_correct", False):
        logger.info("  p  + Enter  — Pause: freeze eval, send sim state to real arm")
        if sync_to_real:
            logger.info("                (sim→real torque OFF → move real arm freely)")
            logger.info("                Real arm drives sim in real-time (real→sim)")
        logger.info("  r           — Resume: continue eval from current sim state")
        if sync_to_real:
            logger.info("                (sim→real torque re-enabled)")
        logger.info("  a           — Abort: discard current episode and move to next")
    if teleop is not None:
        logger.info("  d           — Discard current episode now")
    if args.save_datasets:
        logger.info(f"  [no intervention + success] → {getattr(args, 'eval_dataset_path', 'Datasets/eval')}")
        logger.info(f"  [intervention    + success] → {getattr(args, 'record_dataset_path', 'Datasets/record')}")
    if getattr(args, "mirror_real", False):
        logger.info("  (real arm)  — Drives simulation at every step (mirror_real)")
    if sync_to_real:
        logger.info("  (sim)       — Drives real arm at every step (sync_to_real)")
    logger.info("  Ctrl+C      — Stop evaluation entirely")
    logger.info(_sep)

    for i in range(args.num_episodes):
        logger.info(f"--- Episode {i + 1}/{args.num_episodes} started ---")
        if getattr(args, "enable_pause_correct", False):
            logger.info("  [Hint] Type 'p' + Enter at any time to pause and intervene.")

        # 1. Reset Environment & Policy
        env.reset()
        policy.reset()
        stabilize_garment_after_reset(env, args)

        # 2. Initial Observation (Numpy)
        object_initial_pose = env.get_all_pose() if args.save_datasets else None
        observation_dict = env._get_observations()

        # === 新增：保存top_rgb图片并调用VLM识别类别 ===
        try:
            import imageio
            import numpy as np
            import subprocess
            import sys
            import os

            top_rgb = observation_dict["observation.images.top_rgb"]
            if top_rgb.shape[0] == 3:
                img = np.transpose(top_rgb, (1, 2, 0))
            elif top_rgb.shape[-1] == 3:
                img = top_rgb
            else:
                raise ValueError(f"Unexpected top_rgb shape: {top_rgb.shape}")

            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            abs_path = os.path.abspath("camera_snapshot.png")
            logger.info(f"[VLM] 保存图片 shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()} 路径: {abs_path}")
            imageio.imwrite(abs_path, img)
            logger.info(f"[VLM] 图片已保存到: {abs_path}")

            # 传递环境变量给子进程（注意：不要覆盖环境变量 env，改用 subprocess_env）
            subprocess_env = os.environ.copy()
            subprocess_env.setdefault('DASHSCOPE_API_KEY', os.environ.get('DASHSCOPE_API_KEY', ''))
            subprocess_env.setdefault('OPENAI_API_KEY', os.environ.get('OPENAI_API_KEY', ''))

            result = subprocess.run(
                [
                    sys.executable,
                    'vlm/isaac_vlm_pipeline.py',
                    '--camera_snapshot', 'camera_snapshot.png',
                    '--policy_type', 'lerobot',
                    '--policy_path', str(args.policy_path),
                    '--dataset_root', str(args.dataset_root),
                    '--num_episodes', '1',
                    '--enable_cameras',
                    '--device', args.device,
                    '--only_classify'
                ],
                capture_output=True, text=True,
                env=subprocess_env,
                timeout=30
            )

            logger.info(f"[VLM] Subprocess stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"[VLM] Subprocess stderr: {result.stderr}")

            garment_policy_map = {
                "top_long": "/mnt/lehome-challenge/outputs/train/ckpt/act_top_long/checkpoints/030000/pretrained_model",
                "pant_long": "/mnt/lehome-challenge/outputs/train/ckpt/act_pant_long/checkpoints/030000/pretrained_model",
                "top_short": "/mnt/lehome-challenge/outputs/train/ckpt/act_top_short/checkpoints/015000/pretrained_model",
                "pant_short": "/mnt/lehome-challenge/outputs/train/ckpt/act_pant_short/checkpoints/035000/pretrained_model",
            }
            default_policy_path = "/mnt/lehome-challenge/outputs/train/act_four_types_0331/checkpoints/060000/pretrained_model"

            output_lines = result.stdout.strip().split('\n')
            garment_type = output_lines[-1] if output_lines else "custom"
            args.garment_type = garment_type

            if garment_type in garment_policy_map:
                args.policy_type = "lerobot"
                candidate_path = garment_policy_map[garment_type]
                config_path = os.path.join(candidate_path, "config.json")
                if os.path.exists(config_path):
                    args.policy_path = candidate_path
                    logger.info(f"[VLM] 识别衣物类别: {garment_type}，使用本地策略路径: {args.policy_path}")
                else:
                    args.policy_path = candidate_path
                    logger.warning(f"[VLM] 识别衣物类别: {garment_type}，但未找到 config.json，直接用目录路径: {args.policy_path}")
            else:
                args.policy_type = "lerobot"
                default_candidate = default_policy_path
                args.policy_path = default_candidate
                logger.warning(f"[VLM] 衣物识别失败或类别未知({garment_type})，使用默认策略路径: {args.policy_path}")

            # Reload policy with the newly determined path
            if policy_kwargs is not None:
                try:
                    new_policy_kwargs = dict(policy_kwargs)
                    new_policy_kwargs["policy_path"] = args.policy_path
                    policy = PolicyRegistry.create(args.policy_type, **new_policy_kwargs)
                    logger.info(f"[VLM] 策略已重新加载: {args.policy_path}")
                except Exception as reload_err:
                    logger.warning(f"[VLM] 策略重新加载失败，继续使用原有策略: {reload_err}")

        except subprocess.TimeoutExpired:
            logger.warning("[VLM] VLM 子进程超时（30秒），使用默认策略")
            args.policy_type = "lerobot"
            args.policy_path = "/mnt/lehome-challenge/outputs/train/act_four_types_0331/checkpoints/060000/pretrained_model"
            logger.warning(f"[VLM] 超时，使用默认策略路径: {args.policy_path}")
            if policy_kwargs is not None:
                try:
                    new_policy_kwargs = dict(policy_kwargs)
                    new_policy_kwargs["policy_path"] = args.policy_path
                    policy = PolicyRegistry.create(args.policy_type, **new_policy_kwargs)
                    logger.info(f"[VLM] 策略已重新加载(超时fallback): {args.policy_path}")
                except Exception as reload_err:
                    logger.warning(f"[VLM] 策略重新加载失败: {reload_err}")

        except Exception as e:
            logger.warning(f"[VLM] 衣物识别失败，使用默认策略，错误信息: {e}")
            args.policy_type = "lerobot"
            args.policy_path = "/mnt/lehome-challenge/outputs/train/act_four_types_0331/checkpoints/060000/pretrained_model"
            if policy_kwargs is not None:
                try:
                    new_policy_kwargs = dict(policy_kwargs)
                    new_policy_kwargs["policy_path"] = args.policy_path
                    policy = PolicyRegistry.create(args.policy_type, **new_policy_kwargs)
                    logger.info(f"[VLM] 策略已重新加载(exception fallback): {args.policy_path}")
                except Exception as reload_err:
                    logger.warning(f"[VLM] 策略重新加载失败: {reload_err}")

        # ---------------------------------------------------------------
        # 准备 episode 循环（注意缩进：现在在 for 循环内部，但不在 try 块内部）
        # ---------------------------------------------------------------
        episode_frames = (
            {k: [] for k in observation_dict.keys() if "images" in k}
            if args.save_video
            else {}
        )

        episode_return = 0.0
        episode_length = 0
        extra_steps = 0
        success_flag = False
        success = torch.tensor(False)
        discard_flag = False
        intervention_flag = False

        if teleop is not None:
            def _on_discard():
                nonlocal discard_flag
                discard_flag = True
                logger.info("[d] Episode will be discarded.")
            teleop.add_callback("D", _on_discard)

        for st in range(args.max_steps):
            if rate_limiter:
                rate_limiter.sleep(env)

            # Pause-correct mode handling (保持不变)
            if getattr(args, "enable_pause_correct", False):
                try:
                    import select
                    import sys
                    if select.select([sys.stdin], [], [], 0)[0]:
                        user_input = sys.stdin.readline().strip().lower()
                        if user_input == 'p':
                            intervention_flag = True
                            logger.info("\n=== PAUSED - Episode paused for human intervention ===")
                            obs = env._get_observations()
                            if "observation.state" in obs:
                                virtual_state = obs["observation.state"]
                                if isinstance(virtual_state, torch.Tensor):
                                    virtual_state_np = virtual_state.cpu().numpy()
                                else:
                                    virtual_state_np = np.array(virtual_state)
                                logger.info(f"Current virtual robot state: {virtual_state_np}")
                                logger.info("Sending state to real robot for intervention...")
                                if teleop is not None:
                                    try:
                                        _write_state_to_real(teleop, virtual_state_np)
                                        logger.info("Sent virtual pose to real robot (converted rad→motor); please verify/correct then press 'r' to resume")
                                    except Exception as e:
                                        logger.warning(f"Could not send to real robot: {e}")
                            resume_flag = False
                            abort_flag = False
                            def on_resume():
                                nonlocal resume_flag
                                resume_flag = True
                                logger.info("[Callback] Resume flag set")
                            def on_abort():
                                nonlocal abort_flag
                                abort_flag = True
                                logger.info("[Callback] Abort flag set")
                            if teleop is not None:
                                teleop.manual_control = True
                                if hasattr(teleop, 'left_so101_leader') and hasattr(teleop, 'right_so101_leader'):
                                    teleop.left_so101_leader._started = True
                                    teleop.right_so101_leader._started = True
                                elif hasattr(teleop, '_started'):
                                    teleop._started = True
                                if sync_to_real:
                                    _disable_real_torque(teleop)
                                    logger.info("Pause: torque disabled — move real arm to correct position (real→sim).")
                                teleop.add_callback("R", on_resume)
                                teleop.add_callback("A", on_abort)
                                logger.info("=== HUMAN INTERVENTION MODE ===")
                                logger.info("  Move real arm freely to correct sim state (real→sim).")
                                logger.info("  Press 'r' to resume evaluation  |  Press 'a' to abort episode.")
                                while not resume_flag and not abort_flag:
                                    teleop_out = teleop.advance()
                                    if teleop_out is None:
                                        time.sleep(0.02)
                                        continue
                                    if isinstance(teleop_out, torch.Tensor):
                                        action = teleop_out.to(args.device)
                                        if action.dim() == 1:
                                            action = action.unsqueeze(0)
                                    elif isinstance(teleop_out, dict) and "joint_state" in teleop_out:
                                        arr = teleop_out["joint_state"]
                                        action_np = arr.cpu().numpy() if isinstance(arr, torch.Tensor) else np.array(arr)
                                        action = torch.from_numpy(action_np).float().to(args.device).unsqueeze(0)
                                    else:
                                        time.sleep(0.02)
                                        continue
                                    if args.use_ee_pose and ee_solver is not None:
                                        current_joints = torch.from_numpy(observation_dict["observation.state"]).float().to(args.device)
                                        action = convert_ee_pose_to_joints(
                                            ee_pose_action=action.squeeze(0),
                                            current_joints=current_joints,
                                            solver=ee_solver,
                                            is_bimanual=is_bimanual,
                                            state_unit="rad",
                                            device=args.device,
                                        ).unsqueeze(0)
                                    env.step(action)
                                    observation_dict = env._get_observations()
                                    if args.save_datasets:
                                        frame = {k: v for k, v in observation_dict.items() if k != "observation.top_depth"}
                                        frame["task"] = args.task_description
                                        eval_dataset.add_frame(frame)
                                        frame["task"] = args.task_description
                                        record_dataset.add_frame(frame)
                                teleop.manual_control = False
                                if sync_to_real and resume_flag:
                                    _enable_real_torque(teleop)
                                    logger.info("Resume: torque re-enabled — arm will follow simulation (sim→real).")
                                logger.info(f"Paused mode exit: resume_flag={resume_flag}, abort_flag={abort_flag}")
                            else:
                                while True:
                                    resume_input = input("Press 'r' to resume, 'a' to abort episode: ").strip().lower()
                                    if resume_input == 'r':
                                        logger.info("Resuming episode...")
                                        break
                                    elif resume_input == 'a':
                                        logger.info("Aborting episode")
                                        st = args.max_steps
                                        abort_flag = True
                                        break
                            if resume_flag:
                                logger.info("Resuming episode...")
                                resume_flag = False
                            if abort_flag:
                                logger.info("Aborting episode")
                                st = args.max_steps
                                break
                except Exception as e:
                    logger.debug(f"Pause-correct check: {e}")

            # Action selection
            if teleop is not None and getattr(args, "mirror_real", False):
                try:
                    teleop_out = teleop.advance()
                    if isinstance(teleop_out, torch.Tensor):
                        action_np = teleop_out.squeeze(0).cpu().numpy()
                    elif isinstance(teleop_out, dict) and "joint_state" in teleop_out:
                        arr = teleop_out["joint_state"]
                        action_np = arr.cpu().numpy() if isinstance(arr, torch.Tensor) else np.array(arr)
                    else:
                        action_np = policy.select_action(observation_dict)
                except Exception:
                    logger.warning("Teleop advance failed in mirror_real mode, defaulting to policy.")
                    action_np = policy.select_action(observation_dict)
            else:
                action_np = policy.select_action(observation_dict)
                if teleop is not None:
                    try:
                        teleop_out = teleop.advance()
                        if isinstance(teleop_out, torch.Tensor):
                            action_np = teleop_out.squeeze(0).cpu().numpy()
                        elif isinstance(teleop_out, dict) and "joint_state" in teleop_out:
                            arr = teleop_out["joint_state"]
                            action_np = arr.cpu().numpy() if isinstance(arr, torch.Tensor) else np.array(arr)
                    except Exception:
                        logger.warning("Teleop advance failed, using policy output.")

            action = torch.from_numpy(action_np).float().to(args.device).unsqueeze(0)

            if args.use_ee_pose and ee_solver is not None:
                current_joints = torch.from_numpy(observation_dict["observation.state"]).float().to(args.device)
                action = convert_ee_pose_to_joints(
                    ee_pose_action=action.squeeze(0),
                    current_joints=current_joints,
                    solver=ee_solver,
                    is_bimanual=is_bimanual,
                    state_unit="rad",
                    device=args.device,
                ).unsqueeze(0)

            env.step(action)

            if not success_flag:
                success = env._get_success()
                if success.item():
                    success_flag = True
                    extra_steps = 50

            reward_value = env._get_rewards()
            reward = reward_value.item() if isinstance(reward_value, torch.Tensor) else float(reward_value)
            episode_return += reward
            if not success_flag:
                episode_length += 1

            observation_dict = env._get_observations()

            if sync_to_real and not getattr(teleop, 'manual_control', False):
                state = observation_dict.get("observation.state")
                if state is not None:
                    state_np = state.cpu().numpy() if isinstance(state, torch.Tensor) else np.array(state)
                    _write_state_to_real(teleop, state_np)

            if args.save_datasets:
                frame = {k: v for k, v in observation_dict.items() if k != "observation.top_depth"}
                frame["task"] = args.task_description
                eval_dataset.add_frame(frame)
                frame["task"] = args.task_description
                record_dataset.add_frame(frame)

            if args.save_video:
                for key, val in observation_dict.items():
                    if "images" in key:
                        episode_frames[key].append(val.copy())

            if discard_flag:
                logger.info("Episode discarded by user (d key).")
                break

            if success_flag:
                extra_steps -= 1
                if extra_steps <= 0:
                    break

        # --- End of Episode Handling ---
        is_success = success.item() if success_flag else False

        if args.save_datasets:
            def _save_episode(dataset, json_path, episode_index, label):
                dataset.save_episode()
                scale = None
                if hasattr(env, "object") and hasattr(env.object, "init_scale"):
                    try:
                        scale = env.object.init_scale
                    except Exception:
                        pass
                append_episode_initial_pose(
                    json_path, episode_index, object_initial_pose,
                    garment_name=garment_name, scale=scale,
                )
                logger.info(f"Episode saved → {label} (index {episode_index}).")
                return episode_index + 1

            if discard_flag:
                eval_dataset.clear_episode_buffer()
                record_dataset.clear_episode_buffer()
                logger.info("Episode discarded (d key).")
            elif success_flag:
                if intervention_flag:
                    episode_index_record = _save_episode(
                        record_dataset, json_path_record, episode_index_record,
                        f"record/{episode_index_record}"
                    )
                    eval_dataset.clear_episode_buffer()
                else:
                    episode_index_eval = _save_episode(
                        eval_dataset, json_path_eval, episode_index_eval,
                        f"eval/{episode_index_eval}"
                    )
                    record_dataset.clear_episode_buffer()
            else:
                eval_dataset.clear_episode_buffer()
                record_dataset.clear_episode_buffer()
                logger.info("Episode not saved (no success).")

        if args.save_video:
            save_videos_from_observations(
                episode_frames,
                success=success if success_flag else torch.tensor(False),
                save_dir=args.video_dir,
                episode_idx=i,
            )

        all_episode_metrics.append(
            {"return": episode_return, "length": episode_length, "success": is_success}
        )
        logger.info(
            f"Episode {i + 1}/{args.num_episodes}: Return={episode_return:.2f}, Length={episode_length}, Success={is_success}"
        )

    # 所有 episode 结束后，清理和返回
    if sync_to_real:
        _disable_real_torque(teleop)
        logger.info("sync_to_real done: real robot torque disabled.")

    if args.save_datasets:
        import shutil as _shutil
        for ds, label in [(eval_dataset, "eval"), (record_dataset, "record")]:
            if ds is None:
                continue
            ds.clear_episode_buffer()
            ds.finalize()
            if ds.meta.total_episodes == 0:
                _shutil.rmtree(ds.root, ignore_errors=True)
                logger.info(f"{label} dataset had 0 saved episodes — directory removed: {ds.root}")
            else:
                logger.info(f"{label} dataset finalized ({ds.meta.total_episodes} episodes) → {ds.root}")

    return all_episode_metrics
def eval(args: argparse.Namespace, simulation_app: Any) -> None:
    """
    Main entry point for evaluation logic.
    """
    # 1. Environment Configuration
    env_cfg = parse_env_cfg(args.task, device=args.device)
    env_cfg.sim.use_fabric = False
    if args.use_random_seed:
        env_cfg.use_random_seed = True
    else:
        env_cfg.use_random_seed = False
        env_cfg.seed = args.seed
        # Propagate seed to sim config if structure exists
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "seed"):
            env_cfg.sim.seed = args.seed

    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path

    # 2. Initialize Policy (Using the Policy Registry)
    # This replaces create_il_policy, make_pre_post_processors, etc.
    logger.info(f"Initializing Policy Type: {args.policy_type}")

    # Check if policy is registered
    if not PolicyRegistry.is_registered(args.policy_type):
        available_policies = PolicyRegistry.list_policies()
        raise ValueError(
            f"Policy type '{args.policy_type}' not found in registry. "
            f"Available policies: {', '.join(available_policies)}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_bimanual = "Bi" in args.task or "bi" in args.task.lower()

    # Create policy instance from registry with appropriate arguments
    # Different policies may require different initialization arguments
    policy_kwargs = {
        "device": device,
    }

    if args.policy_type == "lerobot":
        # LeRobot policy requires policy_path and dataset_root
        if not args.policy_path:
            raise ValueError("--policy_path is required for lerobot policy type")
        if not args.dataset_root:
            raise ValueError("--dataset_root is required for lerobot policy type")
        policy_kwargs.update(
            {
                "policy_path": args.policy_path,
                "dataset_root": args.dataset_root,
                "task_description": args.task_description,
            }
        )
    else:
        # For custom policies, pass policy_path as model_path if provided
        if args.policy_path:
            policy_kwargs["model_path"] = args.policy_path

    # Create policy from registry
    policy = PolicyRegistry.create(args.policy_type, **policy_kwargs)
    logger.info(f"Policy '{args.policy_type}' loaded successfully")

    # 3. Initialize IK Solver (If needed)
    ee_solver = None
    if args.use_ee_pose:
        from lehome.utils import RobotKinematics

        urdf_path = args.ee_urdf_path  # Assuming path is handled or add check logic
        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        ee_solver = RobotKinematics(
            str(urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=joint_names,
        )
        logger.info(f"IK solver loaded.")

    # 4. Load Evaluation List
    # Only loads from 'Release' directory based on garment_type
    eval_list = []  # List of (name, stage)

    # Evaluate a specific category based on garment_type
    if args.garment_type == "custom":
        # For 'custom' type, we load from the root Release_test_list.txt
        eval_list_path = os.path.join(
            args.garment_cfg_base_path, "Release", "Release_test_list.txt"
        )
    else:
        # Map argument to specific sub-category directory
        type_map = {
            "top_long": "Top_Long",
            "top_short": "Top_Short",
            "pant_long": "Pant_Long",
            "pant_short": "Pant_Short",
        }
        file_prefix = type_map.get(args.garment_type, "Top_Long")
        # Path: Assets/objects/Challenge_Garment/Release/Top_Long/Top_Long.txt
        eval_list_path = os.path.join(
            args.garment_cfg_base_path, "Release", file_prefix, f"{file_prefix}.txt"
        )

    logger.info(
        f"Loading evaluation list for category '{args.garment_type}' from: {eval_list_path}"
    )

    if not os.path.exists(eval_list_path):
        raise FileNotFoundError(f"Evaluation list not found: {eval_list_path}")

    with open(eval_list_path, "r") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
        for name in names:
            eval_list.append((name, "Release"))

    logger.info(f"Loaded {len(eval_list)} garments for category: {args.garment_type}")

    if not eval_list:
        raise ValueError(
            f"No garments found to evaluate for category '{args.garment_type}'."
        )

    # 5. Main Evaluation Loops
    all_garment_metrics = []

    # Init Env with first garment
    first_name, first_stage = eval_list[0]
    env_cfg.garment_name = first_name
    env_cfg.garment_version = first_stage
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.initialize_obs()

    try:
        for garment_idx, (garment_name, garment_stage) in enumerate(eval_list):
            logger.info(
                f"Evaluating: {garment_name} ({garment_stage}) ({garment_idx+1}/{len(eval_list)})"
            )

            # Switch Garment Logic
            if garment_idx > 0:
                if hasattr(env, "switch_garment"):
                    env.switch_garment(garment_name, garment_stage)
                    env.reset()
                    policy.reset()
                else:
                    env.close()
                    env_cfg.garment_name = garment_name
                    env_cfg.garment_version = garment_stage
                    env = gym.make(args.task, cfg=env_cfg).unwrapped
                    env.initialize_obs()
                    policy.reset()

            # Run Loop
            metrics = run_evaluation_loop(
                env=env,
                policy=policy,
                args=args,
                ee_solver=ee_solver,
                is_bimanual=is_bimanual,
                garment_name=garment_name,
                policy_kwargs=policy_kwargs,
            )

            all_garment_metrics.append(
                {"garment_name": garment_name, "metrics": metrics}
            )

    finally:
        env.close()

    # Print summary across all garments
    logger.info("=" * 60)
    logger.info("Overall Summary")
    logger.info("=" * 60)

    if all_garment_metrics:
        # Aggregate all episode metrics
        all_episodes = []
        for garment_data in all_garment_metrics:
            for episode_metric in garment_data["metrics"]:
                episode_metric["garment_name"] = garment_data["garment_name"]
                all_episodes.append(episode_metric)

        # Print overall metrics
        calculate_and_print_metrics(all_episodes)

        # Print per-garment summary
        logger.info("=" * 60)
        logger.info("Per-Garment Summary")
        logger.info("=" * 60)
        for garment_data in all_garment_metrics:
            garment_name = garment_data["garment_name"]
            metrics = garment_data["metrics"]
            success_count = sum(1 for m in metrics if m["success"])
            success_rate = success_count / len(metrics) if metrics else 0.0
            avg_return = np.mean([m["return"] for m in metrics]) if metrics else 0.0
            logger.info(
                f"  {garment_name}: Success Rate = {success_rate:.2%}, Avg Return = {avg_return:.2f}"
            )
    else:
        logger.info("No metrics collected (all evaluations failed)")

    logger.info("=" * 60)
    logger.info("Evaluation completed successfully")
    logger.info("=" * 60)
