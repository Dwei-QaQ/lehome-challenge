import dashscope
from dashscope import MultiModalConversation
import os

# 设置API Key（推荐从环境变量读取）
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', 'sk-你的真实Key')

# 你的本地图片路径
local_image_path = '/mnt/lehome-challenge/camera_snapshot.png'  # 修改为实际路径

messages = [{
    "role": "user",
    "content": [
        {"image": local_image_path},          # 本地图片
        {"text": "这张图中是什么类型的上衣？例如：短袖T恤、长袖衬衫、外套、卫衣等。"}
    ]
}]

response = MultiModalConversation.call(
    model='qwen-vl-max',    # 也可以换成 'qwen-vl-plus'
    messages=messages
)

if response.status_code == 200:
    print("识别结果:", response.output.choices[0].message.content)
else:
    print(f"错误: {response.code} - {response.message}")