import torch
from typing import List, Union
from transformers import (
    Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
)
import re
from qwen_vl_utils import process_vision_info
from jinja2 import Template

with open('/data/user0/PycharmProjects/EasyR1-main/examples/format_prompt/noise_judge.jinja', encoding="utf-8") as f:
    prompt_template = f.read()
prompt = Template(prompt_template.strip())

def parse_predicted_order(output_text, image_list):
    """解析模型输出，返回 predicted_order 或 None"""
    match = re.search(r'<answer>\[(.*?)\]</answer>', output_text)
    if not match:
        return None

    try:
        tmp_predicted_order = []
        predicted_order = [int(x) - 1 for x in match.group(1).strip().split(',') if x.strip()]

        for idx in predicted_order:
            if 0 <= idx < len(image_list):
                tmp_predicted_order.append(idx)

        return tmp_predicted_order

    except Exception as e:
        print(f"Parsing error: {str(e)}, output text: {output_text}")
        return None

class QueryNoiser:
    """
    Universal query reranker that supports any model for image reranking
    """
    
    def __init__(self, model_path: str, device):
        """
        Initialize the reranker
        
        Args:
            model_path: Model path
        """
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            # device_map="auto",
        )
            
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        self.model.to(device)
        
    def find_noise(self, query: str, image_list: List[str], max_retries=3) -> List[int]:
        """
        Rerank query results
        
        Args:
            query: Query text
            image_list: List of image paths
            
        Returns:
            List[int]: Reranked index list
        """
        device = self.model.device

        prompt_str = prompt.render(question=query, images=image_list)
        content_list = []
        img_idx = 0
        # 按 <image> 进行切分
        for i, text_part in enumerate(prompt_str.split("<image>")):
            # 从第二段开始，先插入一张图片
            if i != 0:
                image_path = image_list[img_idx]
                content_list.append({
                    "type": "image",
                    "image": image_path,
                })
                img_idx += 1

            # 再插入这段文本（如果非空）
            if text_part:
                content_list.append({
                    "type": "text",
                    "text": text_part,
                })

        messages = [
            {
                "role": "user",
                "content": content_list
            }
        ]
            
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(device)

        for attempt in range(1, max_retries + 1):
            generated_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=0.3,
                max_new_tokens=8192,
                use_cache=True,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # === 3. 尝试解析 ===
            predicted_order = parse_predicted_order(output_text, image_list)

            if predicted_order is not None:
                return predicted_order  # 成功解析，直接返回

            print(f"Retry {attempt}/{max_retries} failed. Retrying...")
        return []
