from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import os
from utils.common_utils import post_process_image
from PIL import Image
import re

# -------------------------
# 1) 加载 vLLM 引擎 + 处理器
# -------------------------
def load_model(model_name="/data/user0/models/Qwen/Qwen2.5-VL-7B-Instruct"):
    llm = LLM(  # vLLM 引擎
        model=model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 10},
        tensor_parallel_size=1,      # 多卡可改
        disable_mm_preprocessor_cache=True,
        enforce_eager=False)
    processor = AutoProcessor.from_pretrained(model_name)  # 只用于模板与tokenizer
    return llm, processor

def to_vllm_images(image_list):
    """把路径列表转换为 PIL.Image 列表，供 vLLM 使用。"""
    if not image_list:
        return []
    if not isinstance(image_list, list):
        image_list = [image_list]
    return [
        post_process_image(
            img.convert("RGB") if isinstance(img, Image.Image)
            else Image.open(os.path.abspath(img)).convert("RGB")
        )
        for img in image_list
    ]

# -------------------------
# 3) 构造 chat 模板
# -------------------------
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
            content.append({"type":"image","image": {"url": u}})
    if content:
        messages.append({"role":"user", "content": content})
    return messages

# -------------------------
# 4) vLLM 推理（支持文本 & Judge）
# -------------------------
def get_yes_no_score_v1(out):
    """从 vLLM 输出中提取 Yes / No 对数概率差值"""
    if not out.logprobs:
        return 0.0

    step_logprobs = out.logprobs[0]  # 只取第一个生成步
    logp_yes = logp_no = None

    for info in step_logprobs.values():
        token = info.decoded_token
        if token == "Yes":
            logp_yes = info.logprob
        elif token == "No":
            logp_no = info.logprob

    if logp_yes and logp_no:
        return logp_yes - logp_no
    else:
        return 0.0

def get_yes_no_score(out):
    """从最终 \\boxed{} 中的 Yes/No 提取对数概率差值 (log P(Yes) - log P(No))"""
    if not out.logprobs:
        return 0.0

    # 定义匹配正则
    yes_pattern = re.compile(r"^\s*yes\s*$", re.IGNORECASE)
    no_pattern = re.compile(r"^\s*no\s*$", re.IGNORECASE)

    for step_logprobs in reversed(out.logprobs):
        yes_logprobs = []
        no_logprobs = []
        for info in step_logprobs.values():
            t = info.decoded_token
            if yes_pattern.match(t):
                yes_logprobs.append(info.logprob)
            elif no_pattern.match(t):
                no_logprobs.append(info.logprob)
        # 如果两类都有候选，取最大 logprob 的差
        if yes_logprobs and no_logprobs:
            logp_yes = max(yes_logprobs)
            logp_no = max(no_logprobs)
            return logp_yes - logp_no
    return 0.0

def mllm_response(mllm: LLM, processor, sys_prompt, user_prompt, image_list, max_new_tokens=512):
    inputs = []
    if isinstance(sys_prompt, list):
        for s,u,i in zip(sys_prompt, user_prompt, image_list):
            m = build_messages(s, u, i)
            text = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            image_urls = to_vllm_images(i)
            inputs.append({"prompt": text, "multi_modal_data": {"image": image_urls}})
    else:
        messages = build_messages(sys_prompt, user_prompt, image_list)
        # 用 tokenizer 的 chat template 生成纯文本 prompt
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # vLLM 多模态：把图片以 URLs 传入（file:// 或 data:URL 都行）
        image_urls = to_vllm_images(image_list)
        inputs.append({"prompt": text, "multi_modal_data": {"image": image_urls}})

    # 普通生成
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    outs = mllm.generate(inputs, sp)
    return [o.outputs[0].text for o in outs]
