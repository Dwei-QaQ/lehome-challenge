import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher

from .utils import common
from .utils.parser import setup_eval_parser
from .utils.common import launch_app_from_args
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Evaluation entrypoint with HIL support.

    This script mirrors the structure of `scripts/eval.py` but adds optional
    teleop/HIL flags. At runtime it monkeypatches `create_teleop_interface`
    from `scripts.utils.dataset_record` to return a wrapped teleop that can
    apply delay/offset before forwarding to the evaluation loop.
    """
    parser = setup_eval_parser()
    AppLauncher.add_app_launcher_args(parser)

    # teleop / HIL flags (non-destructive; only used when provided)
    parser.add_argument(
        "--teleop_device",
        type=str,
        default=None,
        choices=[None, "keyboard", "bi-keyboard", "so101leader", "bi-so101leader"],
        help="Optional teleop device for manual override during evaluation.",
    )
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--left_arm_port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--right_arm_port", type=str, default="/dev/ttyACM1")
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        default=False,
        help="Recalibrate teleop device on startup.",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="Sensitivity scale for keyboard teleop.",
    )
    parser.add_argument(
        "--mirror_real",
        action="store_true",
        help="When set with --teleop_device, drive sim from real robot joint states.",
    )
    parser.add_argument(
        "--mirror_delay",
        type=int,
        default=0,
        help="Delay in steps to apply to teleop joint_state (FIFO).",
    )
    parser.add_argument(
        "--joint_offset",
        type=float,
        nargs="+",
        default=None,
        help="Per-joint offset applied to teleop joint_state.",
    )
    parser.add_argument(
        "--enable_pause_correct",
        action="store_true",
        help="Enable pause-correct mode: pause sim, sync state to real robot, allow human intervention",
    )

    args = parser.parse_args()
    simulation_app = launch_app_from_args(args)

    try:
        # Import lehome tasks to register them in gymnasium (must be after simulation_app init)
        import lehome.tasks  # noqa: F401
        
        import scripts.utils.dataset_record as dr_mod
        # lazy import TeleopWrapper implemented in hil_eval_wrapper to avoid duplication
        try:
            from scripts.hil_eval_wrapper import TeleopWrapper
        except Exception:
            TeleopWrapper = None

        orig_factory = getattr(dr_mod, "create_teleop_interface", None)

        if orig_factory is not None and args.teleop_device is not None:
            def factory_wrapper(env, call_args):
                base = orig_factory(env, call_args)
                if getattr(args, "mirror_real", False) or getattr(call_args, "mirror_real", False):
                    # apply delay/offset if TeleopWrapper available, otherwise return base
                    if TeleopWrapper is not None:
                        return TeleopWrapper(base, delay=args.mirror_delay, offset=args.joint_offset)
                    else:
                        return base
                return base

            dr_mod.create_teleop_interface = factory_wrapper

        # run normal eval (uses patched factory when teleop_device provided)
        from .utils.evaluation import eval as eval_main

        # Verify task is a complete environment ID before evaluation
        if hasattr(args, 'task') and args.task:
            import gymnasium as gym
            try:
                # Try to get spec; if it fails, error will be raised properly
                gym.spec(args.task)
                logger.info(f"Task '{args.task}' is valid, starting evaluation...")
            except Exception as e:
                logger.error(f"Task '{args.task}' is not a valid environment ID: {e}")
                logger.info("Available LeHome environments: LeHome-SO101-Direct-Garment-v0, LeHome-BiSO101-Direct-Garment-v0, LeHome-BiSO101-Direct-Garment-v2")
                raise

        eval_main(args, simulation_app)

    except Exception as e:
        logger.error(f"Error during HIL evaluation: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # restore factory if it was patched
        try:
            if orig_factory is not None:
                dr_mod.create_teleop_interface = orig_factory
        except Exception:
            pass
        common.close_app(simulation_app)


if __name__ == "__main__":
    main()
