import base64
import json
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Any

from openai import OpenAI

API_KEY = os.getenv('DASHSCOPE_API_KEY')

MODEL_NAME = "qwen3.6-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

ALLOWED_LABELS = ["top_long", "top_short", "pant_long", "pant_short"]
REFERENCE_DIR = Path(__file__).resolve().parent / "reference_images"

LABEL_FEATURES = """
top_long: Upper-body garment with two full-length long sleeves extending from the torso, covering the entire arm. Visual features: distinct torso section + two long sleeve tubes, typical styles include long-sleeve shirts, hoodies, jackets, flannel shirts.
top_short: Upper-body garment with short sleeves (only covering shoulder/upper arm) or sleeveless design. Visual features: torso section with short/absent sleeve tubes, typical styles include t-shirts, tank tops, short-sleeve shirts.
pant_long: Lower-body garment with two full-length long pant legs extending from the waist, covering the entire leg down to the ankle/foot. Visual features: two long pant tubes with hems near the ankle, typical styles include jeans, sweatpants, casual long pants.
pant_short: Lower-body garment with two short pant legs ending above the knee/mid-thigh, not reaching the ankle. Visual features: two short pant tubes with hems high on the leg, typical styles include denim shorts, athletic shorts, casual shorts.
""".strip()

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def _to_data_url(image_path: str) -> str:
    # 将本地图片转成 data URL，便于直接放进多模态请求中，
    # 不需要额外上传到图床或文件服务。
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "application/octet-stream"
    image_bytes = path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _to_data_url_bytes(image_bytes: bytes, filename: str = "image.png") -> str:
    mime_type, _ = mimetypes.guess_type(filename)
    mime_type = mime_type or "application/octet-stream"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _to_data_url_numpy(image_array: Any, filename: str = "image.png") -> str:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is required to convert numpy arrays to image data URLs. "
            "Install numpy and retry."
        ) from exc

    array = np.asarray(image_array)
    if array.ndim == 2:
        mode = "L"
    elif array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
        mode = "L"
    elif array.ndim == 3 and array.shape[2] == 3:
        mode = "RGB"
    elif array.ndim == 3 and array.shape[2] == 4:
        mode = "RGBA"
    else:
        raise TypeError(
            f"Unsupported image array shape: {array.shape}. "
            "Expected HxW, HxWx1, HxWx3, or HxWx4."
        )

    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            if array.max() <= 1.0:
                array = np.clip(array, 0.0, 1.0) * 255.0
            else:
                array = np.clip(array, 0.0, 255.0)
        else:
            array = np.clip(array, 0, 255)
        array = array.astype(np.uint8)

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to convert numpy arrays to image data URLs. "
            "Install pillow and retry."
        ) from exc

    image = Image.fromarray(array, mode)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return _to_data_url_bytes(buffer.getvalue(), filename)


def _image_to_data_url(image: Any, filename: str = "image.png") -> str:
    if isinstance(image, (bytes, bytearray)):
        return _to_data_url_bytes(bytes(image), filename)
    if isinstance(image, str):
        return _to_data_url(image)
    return _to_data_url_numpy(image, filename)


def _build_reference_content() -> list[dict]:
    # 从专门的参考图目录中为每个标签加载一张样例图，
    # 帮助模型把标签名称和视觉特征对应起来。
    content: list[dict] = []

    for label in ALLOWED_LABELS:
        reference_path = None
        for suffix in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = REFERENCE_DIR / f"{label}{suffix}"
            if candidate.is_file():
                reference_path = candidate
                break

        if reference_path is None:
            continue

        content.append({"type": "text", "text": f"Reference image for label {label}:"})
        content.append(
            {"type": "image_url", "image_url": {"url": _to_data_url(str(reference_path))}}
        )

    return content


def classify_clothing_label(image: Any, filename: str = "image.png") -> dict[str, str]:
    image_data_url = _image_to_data_url(image, filename)
    reference_content = _build_reference_content()

    # 同时使用文字定义和参考样例图进行分类，并强制模型返回 JSON，
    # 方便后续直接解析结果。
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clothing classifier. "
                    f"Allowed labels: {ALLOWED_LABELS}. "
                    f"Classification features:\n{LABEL_FEATURES}\n"
                    'Return JSON only, like {"label":"top_long"}.'
                ),
            },
            {
                "role": "user",
                "content": reference_content
                + [
                    {
                        "type": "text",
                        "text": (
                            "The images above are labeled reference examples. "
                            "Use them together with the feature definitions below to classify the next query image."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {
                        "type": "text",
                        "text": "Classify the query clothing image and return only one JSON field: label.",
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
    )

    result = json.loads(completion.choices[0].message.content)
    label = str(result.get("label", "")).strip()
    # 只接受预设标签集合中的结果，避免模型返回不支持的标签值。
    if label not in ALLOWED_LABELS:
        raise ValueError(f"Unexpected label: {label!r}")
    return {"label": label}
