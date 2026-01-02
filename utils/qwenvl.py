from transformers import AutoProcessor
import torch
from qwen_vl_utils import process_vision_info
from utils.common_utils import encode_image


def load_model(model_name="/data/user0/models/Qwen/Qwen2.5-VL-7B-Instruct", device = 'cuda:0'):
    if "Qwen2.5" in model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration
        mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    else:
        from transformers import Qwen2VLForConditionalGeneration
        mllm = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    processor = AutoProcessor.from_pretrained(model_name)
    mllm.to(device)
    return mllm, processor

def build_messages(sys_prompt, user_prompt, image_list):
    messages = []
    if sys_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": sys_prompt}],
        })
    if user_prompt or image_list:
        content = []
        if user_prompt:
            content.append({"type": "text", "text": user_prompt})
        if image_list:
            if not isinstance(image_list, list):
                image_list = [image_list]
            image_base64_list = [encode_image(i) for i in image_list]
            for base64_img in image_base64_list:
                content.append({
                    "type": "image",
                    "image": base64_img
                })
        messages.append({
            "role": "user",
            "content": content,
        })
    return messages

def mllm_response(mllm, processor, sys_prompt, user_prompt, image_list, max_new_tokens=512):
    texts, image_inputs = [], []
    if isinstance(sys_prompt, list):
        for s,u,i in zip(sys_prompt, user_prompt, image_list):
            messages = build_messages(s, u, i)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_input, _ = process_vision_info(messages)
            texts.append(text)
            image_inputs.append(image_input)
    else:
        messages = build_messages(sys_prompt, user_prompt, image_list)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_input, _ = process_vision_info(messages)
        texts.append(text)
        image_inputs.append(image_input)
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(mllm.device)

    outputs = mllm.generate(**inputs, return_dict_in_generate=True, output_logits=True, max_new_tokens=max_new_tokens)

    generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
    output_text = processor.decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return [output_text]