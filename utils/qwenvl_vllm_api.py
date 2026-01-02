from openai import OpenAI
import os
from PIL import Image
from utils.common_utils import post_process_image, encode_image

def get_client(api_key='EMPTY', base_url='http://localhost:8001/v1'):
    client = OpenAI(
        # 将这里换成你在便携AI聚合API后台生成的令牌
        api_key=api_key,
        # 这里将官方的接口访问地址替换成便携AI聚合API的入口地址
        base_url=base_url,
    )
    return client

def build_messages(sys_prompt, user_prompt, image_list):
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    content = []
    if user_prompt:
        content.append({"type":"text","text": user_prompt})
    if image_list:
        if not isinstance(image_list, list):
            image_list = [image_list]
        for u in image_list:
            content.append({"type": "image_url", "image_url": {"url": encode_image(post_process_image(
                u.convert("RGB") if isinstance(u, Image.Image)
                else Image.open(os.path.abspath(u)).convert("RGB")
            ))}})
    if content:
        messages.append({"role":"user", "content": content})
    return messages

def infer(client, sys_prompt, user_prompt, image_list, max_retries=5, model='Qwen-VL'):
    messages = build_messages(sys_prompt, user_prompt, image_list)
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, timeout=100  # type: ignore
            )
            content = response.choices[0].message.content
            return content
        except:
            pass
    return None