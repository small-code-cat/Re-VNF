import os
import torch
from jinja2 import Template
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

from utils.common_utils import process_image, get_tokenizer


def _get_logit_bias(processor):
    # 禁止输出 image token，不改变原逻辑
    if processor is not None and hasattr(processor, "image_token"):
        tid = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {tid: -100}
    return None


def _process_multi_modal_data(multi_modal_data, min_pixels, max_pixels):
    """原逻辑简化版：返回 {'image':[...]} 或 {'video':[...]}"""
    images, videos = [], []

    if "images" in multi_modal_data:
        for img in multi_modal_data["images"]:
            images.append(process_image(img, min_pixels, max_pixels))
    if images:
        return {"image": images}
    return None


# -------------------------------
#   ✨ 最简化的 vLLM 多模态调用类
# -------------------------------
class SimpleVLLM:
    def __init__(self, model_path, jinja_path='/data/user0/PycharmProjects/EasyR1-main/examples/format_prompt/noise_judge.jinja'):
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            trust_remote_code=True,
            tensor_parallel_size=1,
            limit_mm_per_prompt={"image": 10},
            gpu_memory_utilization=0.6,
            disable_mm_preprocessor_cache=True,
            enforce_eager=False,
            max_model_len=21000
        )

        # sampling 参数保留原来结构的简化版
        self.sampling_params = SamplingParams(
            max_tokens=2048,
            temperature=0.0
        )
        with open(jinja_path, encoding="utf-8") as f:
            prompt_template = f.read()
        self.format_prompt = Template(prompt_template.strip())
        self.processor = AutoProcessor.from_pretrained(model_path)  # 只用于模板与tokenizer
        self.tokenizer = get_tokenizer(
            model_path,
            trust_remote_code = True,
            use_fast = True,
        )
        self.min_pixels = 262144
        self.max_pixels = 1048576

        print("Initialized Simplified vLLM engine.")

    def _build_messages(self, query, images, system_prompt):
        prompt_str = self.format_prompt.render(question=query, images=images)

        content_list = []
        for i, content in enumerate(prompt_str.split("<image>")):
            if i != 0:
                content_list.append({"type": "image"})

            if content:
                content_list.append({"type": "text", "text": content})

        # 2. 构造 User Message
        messages = [{"role": "user", "content": content_list}]

        # 3. 如果有 System Prompt，插入到最前面
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        return messages

    @torch.no_grad()
    def generate(self, query, images, system_prompt=None):
        """
        prompt_token_ids: list[int]
        multi_modal_data: {"images":[...]} 或 {"videos":[...]}
        meta: {"min_pixels":..., "max_pixels":..., "video_fps":...}
        """
        messages = self._build_messages(query, images, system_prompt)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        processed_images = [] if len(images) != 0 else None  # text-only data
        for image in images:
            processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

        multi_modal_data = {"images": images}
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        vllm_inputs = [{
            "prompt_token_ids": list(raw_prompt_ids),
            "multi_modal_data": _process_multi_modal_data(
                multi_modal_data,
                self.min_pixels,
                self.max_pixels
            ),
        }]

        outs = self.llm.generate(
            prompts=vllm_inputs,
            sampling_params=self.sampling_params,
        )

        return [o.outputs[0].text for o in outs]
