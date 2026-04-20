"""HIL (Hardware-in-Loop) evaluation with bidirectional synchronization.

This module provides evaluation functions that support sending virtual robot
state back to the real hardware, creating a closed-loop HIL system.
"""

import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Any

from isaaclab.envs import DirectRLEnv
from scripts.eval_policy.registry import PolicyRegistry
from scripts.utils.evaluation import (
    stabilize_garment_after_reset,
    append_episode_initial_pose,
    save_videos_from_observations,
    RateLimiter,
)
from lerobot.data_utils.dataset_utils import get_next_experiment_path_with_gap
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


def eval_hil_bidirectional(args: argparse.Namespace, simulation_app: Any, teleop_device: Any = None) -> list:
    """
    HIL evaluation with bidirectional control:
    - Read real robot state from teleop_device
    - Execute in virtual environment
    - Send virtual result back to real robot (if possible)
    
    Args:
        args: Command-line arguments
        simulation_app: IsaacSim application instance
        teleop_device: Optional teleop device instance for bidirectional control
        
    Returns:
        List of episode metrics
    """
    from scripts.utils.common import create_env_from_hydra_cfg
    from scripts.utils.parser import create_policy_kwargs
    from scripts.utils.evaluation import (
        create_ee_solver,
        convert_ee_pose_to_joints,
        get_garment_config,
    )
    
    # Initialize environment
    env_cfg = None
    try:
        from isaaclab_tasks.utils import parse_env_cfg
        env_cfg = parse_env_cfg(args.task, device=args.device)
        env_cfg.sim.use_fabric = False
        if args.use_random_seed:
            env_cfg.use_random_seed = True
        else:
            env_cfg.use_random_seed = False
            env_cfg.seed = args.seed
            if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "seed"):
                env_cfg.sim.seed = args.seed
    except Exception as e:
        logger.error(f"Failed to parse environment config: {e}")
        raise
    
    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env = create_env_from_hydra_cfg(env_cfg, simulation_app)
    
    # Initialize policy
    policy_kwargs = create_policy_kwargs(args)
    policy = PolicyRegistry.create(args.policy_type, **policy_kwargs)
    
    # Setup recording if needed
    eval_dataset = None
    json_path = None
    if args.save_datasets:
        root_path = Path(args.eval_dataset_path)
        eval_dataset = LeRobotDataset.create(
            repo_id="lehome_eval",
            fps=args.step_hz,
            root=get_next_experiment_path_with_gap(root_path),
            use_videos=True,
            image_writer_threads=8,
            image_writer_processes=0,
            features=None,
        )
        json_path = eval_dataset.root / "meta" / "garment_info.json"
    
    all_episode_metrics = []
    logger.info(f"Starting HIL bidirectional evaluation: {args.num_episodes} episodes")
    rate_limiter = RateLimiter(args.step_hz)
    
    is_bimanual = hasattr(env_cfg, "num_envs") and env_cfg.num_envs == 2
    ee_solver = None
    if args.use_ee_pose:
        ee_solver = create_ee_solver(env_cfg.robot_cfg, is_bimanual, args.device)
    
    for i in range(args.num_episodes):
        logger.info(f"\n=== Episode {i + 1}/{args.num_episodes} ===")
        
        # Reset
        env.reset()
        policy.reset()
        stabilize_garment_after_reset(env, args)
        
        episode_return = 0.0
        episode_length = 0
        success_flag = False
        extra_steps = 0
        
        garment_name, garment_config = get_garment_config(env, args)
        object_initial_pose = env._get_object_pos_rot() if args.save_datasets else None
        episode_index = 0
        
        observation_dict = env._get_observations()
        episode_frames = {}
        if args.save_video:
            for key in observation_dict:
                if "images" in key:
                    episode_frames[key] = []
        
        step_count = 0
        while step_count < args.max_steps:
            # Rate limiting
            if rate_limiter:
                rate_limiter.sleep(env)

            # pause-correct support: freeze counter during manual intervention
            if getattr(args, "enable_pause_correct", False):
                try:
                    import select, sys
                    if select.select([sys.stdin], [], [], 0)[0]:
                        inp = sys.stdin.readline().strip().lower()
                        if inp == 'p':
                            logger.info("\n=== PAUSED (HIL) - waiting for intervention ===")
                            # show state to user
                            obs = env._get_observations()
                            vs = obs.get("observation.state")
                            if vs is not None:
                                if isinstance(vs, torch.Tensor):
                                    vnp = vs.cpu().numpy()
                                else:
                                    vnp = np.array(vs)
                                logger.info(f"Virtual state: {vnp}")
                                if teleop_device is not None and hasattr(teleop_device, '_bus') and hasattr(teleop_device._bus, 'write_desired_position'):
                                    try:
                                        teleop_device._bus.write_desired_position(vnp)
                                        logger.info("Sent to real robot; press 'r' to resume")
                                    except Exception as e:
                                        logger.warning(f"HIL pause write failed: {e}")
                            # wait for resume/abort
                            while True:
                                resp = input("Press 'r' to resume, 'a' to abort: ").strip().lower()
                                if resp == 'r':
                                    break
                                elif resp == 'a':
                                    step_count = args.max_steps
                                    break
                except Exception:
                    pass

            # now perform HIL step
            # 1. Read real robot state via teleop device (HIL input)
            if teleop_device is not None:
                try:
                    teleop_out = teleop_device.advance()
                    if isinstance(teleop_out, dict) and "joint_state" in teleop_out:
                        # Real robot state from teleop
                        real_state = teleop_out["joint_state"]
                        if isinstance(real_state, torch.Tensor):
                            real_state_np = real_state.cpu().numpy()
                        else:
                            real_state_np = np.array(real_state)
                        
                        logger.debug(f"Step {step_count}: Received real robot state: {real_state_np}")
                except Exception as e:
                    logger.warning(f"Failed to read real robot state: {e}")
                    real_state_np = None
            else:
                real_state_np = None
            
            # 2. Policy inference
            action_np = policy.select_action(observation_dict)
            action = torch.from_numpy(action_np).float().to(args.device).unsqueeze(0)
            
            # 3. Apply IK if needed
            if args.use_ee_pose and ee_solver is not None:
                current_joints = (
                    torch.from_numpy(observation_dict["observation.state"])
                    .float()
                    .to(args.device)
                )
                action = convert_ee_pose_to_joints(
                    ee_pose_action=action.squeeze(0),
                    current_joints=current_joints,
                    solver=ee_solver,
                    is_bimanual=is_bimanual,
                    state_unit="rad",
                    device=args.device,
                ).unsqueeze(0)
            
            # 4. Step environment
            env.step(action)

            # update count after actual step
            step_count += 1
            
            # 5. Get virtual robot state after step
            observation_dict = env._get_observations()
            virtual_state = observation_dict.get("observation.state")
            
            # 6. Send virtual state back to real robot (HIL output)
            if teleop_device is not None and virtual_state is not None:
                try:
                    if isinstance(virtual_state, torch.Tensor):
                        virtual_state_np = virtual_state.cpu().numpy()
                    else:
                        virtual_state_np = np.array(virtual_state)
                    
                    # Try to send command to real robot
                    # This assumes the real robot has a method to accept joint targets
                    if hasattr(teleop_device, '_bus') and hasattr(teleop_device._bus, 'write_desired_position'):
                        teleop_device._bus.write_desired_position(virtual_state_np)
                        logger.debug(f"Step {step_count}: Sent virtual state to real robot: {virtual_state_np}")
                except Exception as e:
                    logger.debug(f"Could not send state to real robot: {e}")
            
            # Check success
            if not success_flag:
                success = env._get_success()
                if success.item():
                    success_flag = True
                    extra_steps = 50
            
            # Accumulate reward
            reward_value = env._get_rewards()
            if isinstance(reward_value, torch.Tensor):
                reward = reward_value.item()
            else:
                reward = float(reward_value)
            
            episode_return += reward
            if not success_flag:
                episode_length += 1
            
            # Recording
            if args.save_datasets:
                frame = {k: v for k, v in observation_dict.items() if k != "observation.top_depth"}
                frame["task"] = args.task_description
                eval_dataset.add_frame(frame)
            
            if args.save_video:
                for key, val in observation_dict.items():
                    if "images" in key:
                        episode_frames[key].append(val.copy())
            
            if success_flag:
                extra_steps -= 1
                if extra_steps <= 0:
                    break
        
        # End of episode
        is_success = success.item() if success_flag else False
        
        if args.save_datasets:
            if success_flag:
                eval_dataset.save_episode()
                append_episode_initial_pose(
                    json_path,
                    episode_index,
                    object_initial_pose,
                    garment_name=garment_name,
                )
                episode_index += 1
            else:
                eval_dataset.clear_episode_buffer()
        
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
    
    return all_episode_metrics
