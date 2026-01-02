from qwen_vl_utils import smart_resize
from PIL import Image
import io
import base64

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