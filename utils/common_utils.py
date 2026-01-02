from qwen_vl_utils import smart_resize
from PIL import Image
import io
import base64
from typing import Any, Optional, Union
from PIL.Image import Image as ImageObject
from io import BytesIO
from transformers import AutoTokenizer
import math
import re
import time

answer_prompt = '''Based on the content of the document page images retrieved in relation to the user's question, answer the user's question below as clearly and concisely as possible.
Only provide a direct, paragraph-style answer. Do not include any introductions, notes, or extra commentary.
If the retrieved document pages do not contain sufficient information to answer the question, simply respond: "Not answerable."
'''


class RAGMeter:
    def __init__(self, model_path):
        """
        初始化 Tokenizer 用于计数。
        注意：Tokenizer 运行在 CPU 上，不会干扰 vLLM 的 GPU 推理。
        """
        print(f"[RAGMeter] Loading tokenizer from {model_path}...")
        try:
            # fast=True 加速分词，trust_remote_code=True 适配 Qwen
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        except Exception as e:
            print(f"[RAGMeter] ⚠️ Tokenizer load failed: {e}. Will use rough estimate.")
            self.tokenizer = None

        # Qwen2.5-VL 的图片 Token 是动态的。
        # 如果 vLLM 不返回 usage，这里只能给一个估算值。
        # Qwen2.5-VL 处理高分辨率图片时 token 数较多，建议设为 600-1000 之间的平均值
        self.img_token_avg_cost = 600
        self.COSTS = {
            "filter": 308,  # 刚才算出来的
            "rerank": 80,  # Rerank 通常很短
            "gen": 196  # Generate 阶段
        }

    def _count(self, text):
        if not text: return 0
        if self.tokenizer:
            return len(self.tokenizer.encode(str(text), add_special_tokens=False))
        return len(str(text)) // 3  # Fallback

    def measure(self, func, *args, input_text="", num_imgs=0, task_type="gen", output_parser=lambda x: x, **kwargs):
        """
        增加 task_type 参数: 'filter', 'rerank', 'gen'
        """
        # 1. 计时 (不变)
        start_t = time.time()
        result = func(*args, **kwargs)
        latency_ms = (time.time() - start_t) * 1000

        # 2. [修改] Input 统计 = 固定模板 + 图片 + 动态Query
        # 获取该任务类型的固定开销，默认为 100
        base_cost = self.COSTS[task_type]

        # 只需要算 Query 的 token
        query_cost = self._count(input_text)

        # 图片开销
        img_cost = num_imgs * self.img_token_avg_cost

        in_tokens = base_cost + query_cost + img_cost

        # 3. Output 统计 (保持不变，还是算实际生成的)
        try:
            out_content = output_parser(result)
            out_tokens = self._count(out_content)
        except:
            out_tokens = 0

        stats = {
            "latency": round(latency_ms, 2),
            "in_tokens": int(in_tokens),
            "out_tokens": int(out_tokens),
            "total_tokens": int(in_tokens + out_tokens)
        }
        return result, stats

def parse_predicted_order(output_text, image_list):
    """解析模型输出，返回 predicted_order 或 None"""
    think_content = re.search(r'<think>(.*?)</think>', output_text, flags=re.DOTALL)
    match = re.search(r'<answer>\[(.*?)\]</answer>', output_text)
    # 如果没有标签格式，则匹配裸的 [...]
    if not match:
        match = re.search(r'\[(.*?)\]', output_text)

    if not match:
        return None, None

    try:
        tmp_predicted_order = []
        predicted_order = [int(x) - 1 for x in match.group(1).strip().split(',') if x.strip()]

        for idx in predicted_order:
            if 0 <= idx < len(image_list):
                tmp_predicted_order.append(idx)

        return think_content.group(1).strip() if think_content is not None else '', tmp_predicted_order

    except Exception as e:
        print(f"Parsing error: {str(e)}, output text: {output_text}")
        return None, None

def get_tokenizer(model_path: str, **kwargs):
    """Create a huggingface pretrained tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)

    if tokenizer.bos_token == "<bos>" and tokenizer.eos_token == "<eos>":
        # the EOS token in gemma2 & gemma3 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        print("Found gemma model. Set eos_token and eos_token_id to <end_of_turn> and 107.")
        tokenizer.eos_token = "<end_of_turn>"

    if tokenizer.pad_token_id is None:
        print("Pad token is None. Set it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image

# -------------------------
# 2) 图像工具
# -------------------------
def post_process_image(image: Image) -> Image:
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height, width, max_pixels=1024 * 28 * 28
    )
    return image.resize((resized_width, resized_height))

mime_types = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

def encode_image(image):
    # 如果是路径字符串
    if isinstance(image, str):
        ext = image[image.rfind("."):].lower()
        mime_type = mime_types.get(ext, "image/*")
        with open(image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    # 如果是 PIL Image 对象
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        fmt = (image.format or "PNG").upper()  # 无格式时用 PNG
        image.save(buf, format=fmt)
        mime_type = f"image/{fmt.lower()}"
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    raise TypeError("encode_image expects a file path or PIL Image object.")