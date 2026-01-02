from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
from qwen_vl_utils import smart_resize
from PIL import Image
import math

# -------------------------
# 1) 加载 vLLM 引擎 + 处理器
# -------------------------
def load_model(model_name="/data/user0/models/Qwen/Qwen2.5-7B-Instruct", device='cuda:3'):
    llm = LLM(  # vLLM 引擎
        model=model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,      # 多卡可改
        disable_mm_preprocessor_cache=True,
        enforce_eager=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return llm, tokenizer

# -------------------------
# 3) 构造 chat 模板
# -------------------------
def build_messages(sys_prompt, user_prompt):
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    return messages

def llm_response(llm: LLM, tokenizer, sys_prompt, user_prompt, max_new_tokens=512):
    inputs = []
    if isinstance(sys_prompt, list):
        for s,u in zip(sys_prompt, user_prompt):
            m = build_messages(s, u)
            text = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            inputs.append({"prompt": text})
    else:
        messages = build_messages(sys_prompt, user_prompt)
        # 用 tokenizer 的 chat template 生成纯文本 prompt
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs.append({"prompt": text})

    # 普通生成
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    outs = llm.generate(inputs, sp)
    return [o.outputs[0].text for o in outs]
