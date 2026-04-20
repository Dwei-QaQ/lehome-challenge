import argparse
import logging
import subprocess
import sys
import time
import shlex
import tempfile
from pathlib import Path
from typing import Optional

import dashscope
from dashscope import MultiModalConversation

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def launch_isaac_sim(launch_command: Optional[str]) -> Optional[subprocess.Popen]:
    """启动 Isaac Sim"""
    if not launch_command:
        logger.warning("No Isaac Sim launch command provided. Start Isaac Sim manually if needed.")
        return None

    logger.info("Launching Isaac Sim: %s", launch_command)
    cmd = shlex.split(launch_command)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(10)
    logger.info("Isaac Sim launch process started (PID=%s).", process.pid)
    return process


def wait_for_snapshot(snapshot_path: Path, timeout_sec: int = 120, interval_sec: int = 2) -> bytes:
    """等待摄像头快照并获取图像"""
    logger.info("Waiting for camera snapshot at %s", snapshot_path)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if snapshot_path.is_file():
            image_bytes = snapshot_path.read_bytes()
            if is_image_valid(image_bytes):
                logger.info("Found valid camera snapshot: %s", snapshot_path)
                return image_bytes
            logger.debug("Snapshot exists but is not valid yet. Retrying...")
        time.sleep(interval_sec)

    raise TimeoutError(f"Timeout waiting for snapshot file: {snapshot_path}")


def is_image_valid(image_bytes: bytes) -> bool:
    """检查图像是否有效"""
    if not image_bytes:
        return False

    try:
        from PIL import Image
        from io import BytesIO
    except ImportError:
        return True

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                return False
            extrema = image.convert("L").getextrema()
            return extrema != (0, 0)
    except Exception:
        return False


def classify_snapshot(image_bytes: bytes, filename: str = "camera.png") -> str:
    """
    通过 DashScope (通义千问-VL) 对图片进行分类，返回衣物类别字符串。
    返回结果映射为: "top_long", "top_short", "pant_long", "pant_short" 或 "custom"
    """
    print("hello from VLM (classify_snapshot called)", file=sys.stderr)
    logger.info("Classifying camera snapshot with VLM via DashScope...")

    import os
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("No API key found. Please set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable.")
        return "custom"

    dashscope.api_key = api_key

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_file.write(image_bytes)
        tmp_path = tmp_file.name

    try:
        # Chain-of-Thought 提示词
        messages = [{
            "role": "user",
            "content": [
                {"image": tmp_path},
                {"text": (
                    "请仅仅保留图片中衣服的轮廓仔细分析图片中的衣物的轮廓，回答以下问题：\n"
                    "1. 这件衣物是否有领口或类似上衣的结构？\n"
                    "2. 是否有两个对称的袖子？\n"
                    "3. 是否有腰部和两个裤腿？\n"
                    "4. 综合判断它属于以下哪一类：长袖上衣（top_long）、短袖上衣（top_short）、长裤（pant_long）、短裤（pant_short）。\n"
                   "请先输出你的推理过程，然后在最后一行输出类别代码（例如：top_long）。"
                )}
            ]
            #"content": [
            #    {"image": tmp_path},
            #    {"text": (
            #        "请根据下面对于四种类型衣物的描述，判断图片中的衣物属于哪一类。\n"
            #        "top_long: Upper-body garment with two full-length long sleeves extending from the torso, covering the entire arm. Visual features: distinct torso section + two long sleeve tubes, typical styles include long-sleeve shirts, hoodies, jackets, flannel shirts.top_short: Upper-body garment with short sleeves (only covering shoulder/upper arm) or sleeveless design. Visual features: torso section with short/absent sleeve tubes, typical styles include t-shirts, tank tops, short-sleeve shirts.pant_long: Lower-body garment with two full-length long pant legs extending from the waist, covering the entire leg down to the ankle/foot. Visual features: two long pant tubes with hems near the ankle, typical styles include jeans, sweatpants, casual long pants.pant_short: Lower-body garment with two short pant legs ending above the knee/mid-thigh, not reaching the ankle. Visual features: two short pant tubes with hems high on the leg, typical styles include denim shorts, athletic shorts, casual shorts."
             #       "请先输出你的推理过程，然后在最后一行输出类别代码（例如：top_long）。"
            #    )}
            #]
        }]

        response = MultiModalConversation.call(
            model='qwen-vl-max',      # 也可以尝试 'qwen-vl-plus'
            messages=messages
        )

        if response.status_code == 200:
            content = response.output.choices[0].message.content
            logger.info(f"VLM raw response: {content}")

            # 提取文本
            result_text = ""
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        result_text = item["text"]
                        break
            elif isinstance(content, str):
                result_text = content
            else:
                result_text = str(content)

            if not result_text:
                logger.warning("Empty text in VLM response")
                return "custom"

            # 尝试从最后一行或整个文本中提取类别代码
            lines = result_text.strip().split('\n')
            # 优先找最后一行包含关键词
            candidate = lines[-1].strip().lower()
            # 如果最后一行是推理文字，则全文搜索
            if "top_long" in candidate or "top_short" in candidate or "pant_long" in candidate or "pant_short" in candidate:
                pass
            else:
                # 全文搜索
                candidate = result_text.lower()
            
            if "top_short" in candidate:
                 return "top_short"
            elif "top_long" in candidate:
                return "top_long"
            elif "pant_short" in candidate:
                return "pant_short"
            elif "pant_long" in candidate:
                return "pant_long"
            else:
                logger.warning(f"Unrecognized VLM output: {result_text}, falling back to 'custom'")
                return "custom"
        else:
            logger.error(f"VLM API error: {response.code} - {response.message}")
            return "custom"

    except Exception as e:
        logger.error(f"Exception during VLM classification: {e}")
        return "custom"
    finally:
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass
def build_policy_command(args: argparse.Namespace, garment_type: str) -> list[str]:
    """构建policy评估命令"""
    command = [sys.executable, "-m", "scripts.eval"]
    command += ["--policy_type", args.policy_type]
    if args.policy_type == "lerobot":
        if not args.policy_path or not args.dataset_root:
            raise ValueError("--policy_path and --dataset_root are required for lerobot policy evaluation.")
        command += ["--policy_path", str(args.policy_path)]
        command += ["--dataset_root", str(args.dataset_root)]
    command += ["--garment_type", garment_type]
    if args.num_episodes is not None:
        command += ["--num_episodes", str(args.num_episodes)]
    if args.enable_cameras:
        command.append("--enable_cameras")
    if args.device:
        command += ["--device", args.device]
    if args.extra_policy_args:
        command += args.extra_policy_args
    return command


def run_policy_evaluation(args: argparse.Namespace, garment_type: str) -> None:
    """执行策略评估"""
    command = build_policy_command(args, garment_type)
    logger.info("Running policy evaluation: %s", " ".join(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Isaac Sim -> VLM -> policy pipeline")
    parser.add_argument(
        "--launch_command",
        type=str,
        default=None,
        help="Shell command to launch Isaac Sim, e.g. 'python3 -m omni.isaac.kit'.",
    )
    parser.add_argument(
        "--camera_snapshot",
        type=Path,
        default=Path("camera_snapshot.png"),
        help="本地摄像头快照文件路径，Isaac Sim 启动后应将视图写入该文件。",
    )
    parser.add_argument("--camera_snapshot_timeout", type=int, default=120, help="等待快照文件的超时时间（秒）。")
    parser.add_argument("--policy_type", type=str, required=True, choices=["lerobot", "custom"], help="Policy 类型。")
    parser.add_argument("--policy_path", type=Path, default=None, help="LeRobot policy checkpoint 路径。仅在 policy_type=lerobot 时必须。")
    parser.add_argument("--dataset_root", type=Path, default=None, help="评估数据集根目录。仅在 policy_type=lerobot 时必须。")
    parser.add_argument("--num_episodes", type=int, default=1, help="评估 episode 数量。")
    parser.add_argument("--enable_cameras", action="store_true", help="运行 policy 评估时启用摄像头视图。")
    parser.add_argument("--device", type=str, default="cpu", help="推理设备，例如 cpu 或 cuda。")
    parser.add_argument(
        "--extra_policy_args",
        nargs=argparse.REMAINDER,
        help="传递给 policy 评估脚本的额外参数。",
    )
    parser.add_argument("--only_classify", action="store_true", help="仅输出衣物类别并退出，不运行策略评估")
    return parser.parse_args()


def main() -> int:
    import os
    # 调试信息输出到 stderr
    print("VLM pipeline started (main)", file=sys.stderr)

    args = parse_args()

    # 兼容环境变量：如果只有 OPENAI_API_KEY，则复制到 DASHSCOPE_API_KEY
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("DASHSCOPE_API_KEY"):
        os.environ["DASHSCOPE_API_KEY"] = os.environ["OPENAI_API_KEY"]

    # 1. 启动 Isaac Sim（可选）
    isaac_process = launch_isaac_sim(args.launch_command)
    try:
        # 2. 获取摄像头快照
        image_bytes = wait_for_snapshot(args.camera_snapshot, timeout_sec=args.camera_snapshot_timeout)
    except Exception as exc:
        logger.error("Failed to capture camera snapshot: %s", exc)
        if isaac_process:
            isaac_process.terminate()
        return 1

    try:
        # 3. 分类衣服（使用 DashScope）
        garment_type = classify_snapshot(image_bytes, filename=args.camera_snapshot.name)
        logger.info("VLM classification result: %s", garment_type)

        # 如果只需要输出类别，直接打印并退出
        if args.only_classify:
            print(garment_type)   # 仅输出一行类别到 stdout
            if isaac_process:
                isaac_process.terminate()
            return 0
    except Exception as exc:
        logger.error("Failed to classify image: %s", exc)
        if isaac_process:
            isaac_process.terminate()
        return 1

    try:
        # 4. 运行对应策略进行叠衣服任务
        run_policy_evaluation(args, garment_type)
    except subprocess.CalledProcessError as exc:
        logger.error("Policy evaluation failed: %s", exc)
        if isaac_process:
            isaac_process.terminate()
        return exc.returncode
    finally:
        if isaac_process:
            logger.info("Stopping Isaac Sim process.")
            isaac_process.terminate()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())