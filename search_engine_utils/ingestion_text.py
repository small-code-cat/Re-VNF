import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import base64
from PIL import Image
import io
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleDirectoryReader
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
import numpy as np
import re
from utils.qwenvl_vllm import post_process_image

from vl_embedding import VL_Embedding

def parse_doc_and_page(path: str):
    """从文件路径提取文档名和页码"""
    stem = Path(path).stem  # ADOBE_2015_10K_102
    parts = stem.split("_")
    page_number = int(parts[-1]) if parts[-1].isdigit() else None
    doc_id = "_".join(parts[:-1])
    return doc_id, page_number

def parse_page_image(image):
    # 组装 multipart files（注意打开后要关闭）
    opened, files = [], []
    for p in [image]:
        f = open(p, "rb")
        opened.append(f)
        files.append(("files", (os.path.basename(p), f)))  # 简化：不显式传 MIME
    # 表单数据——就是你给的那份，保持默认值
    data = {
        "backend": "vlm-vllm-async-engine",
        "output_dir": '/data/user0/PycharmProjects/VRAG-main/mineru_output',
        "lang_list": ['en'],
        "parse_method": "auto",
        "return_middle_json": False,
        "return_content_list": True
    }

    try:
        resp = requests.post('http://127.0.0.1:9000/file_parse', files=files, data=data, timeout=60 * 60)
        resp.raise_for_status()
        if resp.status_code == 200:
            return resp.json()['results'][os.path.splitext(os.path.basename(image))[0]]['md_content']
        return ''
    finally:
        for f in opened:
            f.close()

def remove_with_optional_newlines(text: str, target: str, new_str='\n\n') -> str:
    """
    从 text 中去掉 target 以及它前后可能存在的换行符或空格
    """
    pattern = rf'(?:\s*\n*)?{re.escape(target)}(?:\s*\n*)?'
    cleaned = re.sub(pattern, lambda _: new_str, text, count=1)
    return cleaned.strip()

def is_valid_table(table) -> bool:
    """
    判断 HTML 表格是否有效：
    - 含有非空单元格文本，或
    - 含有图片、链接、公式等内容
    """
    # 遍历所有单元格
    for cell in table.find_all(["td", "th"]):
        # 提取文字并清理空白符
        text = cell.get_text(strip=True)
        # 如果单元格中有文字、图片、公式或链接，就认为有效
        if text or cell.find(["img", "a", "math"]):
            return True
    return False

def image_to_base64(image_input, format="png") -> str:
    """
    将多种类型的图像输入转换为 base64 编码字符串。

    支持输入：
        - 文件路径（str）
        - PIL.Image.Image
        - bytes（图像二进制）
        - io.BytesIO
        - NumPy 数组（自动转 PIL）
    """
    if isinstance(image_input, str):  # 文件路径
        with open(image_input, "rb") as f:
            image_bytes = f.read()
    elif isinstance(image_input, bytes):
        image_bytes = image_input
    elif isinstance(image_input, io.BytesIO):
        image_bytes = image_input.getvalue()
    elif isinstance(image_input, Image.Image):
        buffer = io.BytesIO()
        image_input.convert("RGB").save(buffer, format=format)
        image_bytes = buffer.getvalue()
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format=format)
        image_bytes = buffer.getvalue()
    else:
        raise TypeError("Unsupported image_input type.")

    return base64.b64encode(image_bytes).decode("utf-8")

def get_mllm_response(prompt, image_path, model='Qwen/Qwen2.5-VL-7B-Instruct'):
    client = OpenAI(
        api_key='EMPTY',
        base_url='http://localhost:8001/v1',
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(post_process_image(Image.open(image_path).convert('RGB')))}"}
                    }
                ]
            }
        ]
    )
    return completion.choices[0].message.content.strip()

def get_description_prompt(image_type):
    # Return different description prompts based on image type
    prompts = {
        "Data visualization": """
        Please provide a detailed description of this data visualization, including:
        1. Chart type (bar/line/pie etc.)
        2. Axis meanings (what x-axis and y-axis represent)
        3. Data trends and key data points
        4. Chart title and legend information
        5. Any notable features or outliers
        """,
        "Flowchart": """
        Please describe this flowchart/diagram in detail, including:
        1. Overall structure and flow direction
        2. Main components/nodes and their relationships
        3. Key decision points or process steps
        4. Any annotations or explanatory text
        """,
        "Infographic": """
        Please describe this infographic in detail, including:
        1. Main theme and sections
        2. Visual elements used (icons/illustrations etc.)
        3. Key data points and statistics
        4. Textual explanations and labels
        """,
        "Natural scene": """
        Please describe this natural scene in detail, including:
        1. Main objects (people/animals/landscape elements)
        2. Environment and background features
        3. Interactions between objects
        4. Colors, lighting and overall atmosphere
        """,
        "Object": """
        Please describe this object/product in detail, including:
        1. Object category and name
        2. Physical attributes (shape/color/material)
        3. Key components or functional elements
        4. Any branding or identification marks
        """,
        "Document": """
        Please describe this document in detail, including:
        1. Document type and title
        2. Main sections
        3. Key content and data
        4. Any signatures, stamps or special markings
        """,
        "Presentation slide": """
        Please describe this PowerPoint slide in detail, including:
        1. Slide title and main message
        2. Structure of content (headings, bullet points, visuals)
        3. Key data or information presented
        4. Design elements (theme colors, fonts, layout)
        5. Any diagrams or charts included
        """,
        "Table": """
        Please describe this table in detail, including:
        1. Table title or caption (if available)
        2. Number of rows and columns, and general layout
        3. Headers of each column and their meanings
        4. Key data points, patterns, or comparisons
        5. Any highlighted cells, merged cells, or special formatting
        6. Notable insights, anomalies, or outliers shown in the data
        """
    }

    # Default prompt if type not found
    default_prompt = "Please describe this image in detail, including main objects, scene, colors and any notable features."

    # Create a mapping from specific types to broader categories
    type_mapping = {
        # Chart types
        "bar": "Data visualization",
        "bar chart": "Data visualization",
        "line": "Data visualization",
        "line chart": "Data visualization",
        "pie": "Data visualization",
        "pie chart": "Data visualization",
        "scatter": "Data visualization",
        "scatter plot": "Data visualization",
        "radar": "Data visualization",
        "radar chart": "Data visualization",
        "histogram": "Data visualization",
        "graph": "Data visualization",

        # Other specific types
        "ppt": "Presentation slide",
        "powerpoint": "Presentation slide",
        "slide": "Presentation slide",
        "diagram": "Flowchart",
        "process": "Flowchart",
        "photo": "Natural scene",
        "picture": "Natural scene",
        "table": "Table",
        "form": "Document",
        "product": "Object"
    }

    # Clean the input type
    cleaned_type = image_type.lower().strip()

    # First check if it's a specific type we've mapped
    for specific_type, broad_category in type_mapping.items():
        if specific_type in cleaned_type:
            return prompts[broad_category]

    # Then try direct matching with broader categories
    for broad_category in prompts:
        if broad_category.lower() in cleaned_type:
            return prompts[broad_category]

    return default_prompt

def get_image_semantics(img_path):
    semantics = {
        "description": None,
        "image_type": None
    }
    max_attempts = 5
    attempts = 0
    while attempts < max_attempts:
        try:
            image_type = get_mllm_response("""
                        Please analyze this image and classify it into one of these categories:
                        - Data visualization (bar chart/line chart/pie chart/scatter plot/radar chart etc.)
                        - Flowchart
                        - Infographic/information visualization
                        - Natural scene (landscape/people/animals etc.)
                        - Object/product
                        - Document
                        - Presentation slide
                        - Table
                        - Other (unclassified)
                        
                        Return only the most matching category name, no explanation needed.
                        """, img_path)
            semantics["image_type"] = image_type
            description_prompt = get_description_prompt(image_type)
            description = get_mllm_response(description_prompt, img_path)
            semantics["description"] = description
            return semantics
        except Exception as e:
            attempts += 1
            semantics["description"] = str(e)
    return semantics

def get_text(input_file):
    _TABLE_RE = re.compile(r'(?is)<table\b[^>]*>.*?</table>')

    stem = Path(input_file).stem
    text = parse_page_image(input_file)
    vlm_path = next(Path('mineru_output').rglob(f"*/{stem}/vlm"))
    with open(next(vlm_path.glob("*_content_list.json")), "r", encoding="utf-8") as f:
        content_list = json.load(f)

    tables = _TABLE_RE.findall(text)
    for tbl in tables:
        soup = BeautifulSoup(tbl, "html.parser")
        table_soup = soup.find("table")
        if not is_valid_table(table_soup):
            text = remove_with_optional_newlines(text, tbl)
            continue
        target_tbl_dict = next((c for c in content_list if c['type'] == 'table' and tbl in c['table_body']), None)
        if not target_tbl_dict:
            continue

        if target_tbl_dict['table_caption'] and not table_soup.find("table_caption"):
            for i in target_tbl_dict['table_caption']:
                text = remove_with_optional_newlines(text, i)
            new_cap = soup.new_tag("table_caption")
            new_cap.string = '\n'.join(target_tbl_dict['table_caption'])
            table_soup.insert(0, new_cap)

        if target_tbl_dict['table_footnote'] and not table_soup.find("table_footnote"):
            for i in target_tbl_dict['table_footnote']:
                text = remove_with_optional_newlines(text, i)
            new_cap = soup.new_tag("table_footnote")
            new_cap.string = '\n'.join(target_tbl_dict['table_footnote'])
            table_soup.insert(0, new_cap)

        if not table_soup.find("caption"):
            new_cap = soup.new_tag("caption")
            semantic = get_image_semantics(str(vlm_path/target_tbl_dict['img_path']))
            new_cap.string = semantic['description']
            table_soup.insert(0, new_cap)
            text = remove_with_optional_newlines(text, tbl, '\n\n' + str(table_soup) + '\n\n')

    pattern = r'!\[([^\]]*)\]\((images/[^\)]+)\)'
    images = re.findall(pattern, text)
    for _, img_path in images:
        target_img_dict = next(c for c in content_list if c['type'] == 'image' and c['img_path'] == img_path)
        soup = BeautifulSoup("", "html.parser")  # 可以是空文档
        # 创建 <image> 标签
        img_tag = soup.new_tag("image")
        # 创建 <caption> 标签
        new_cap = soup.new_tag("image_path")
        new_cap.string = img_path
        # 嵌入 caption
        img_tag.append(new_cap)

        if target_img_dict['image_caption']:
            for i in target_img_dict['image_caption']:
                text = remove_with_optional_newlines(text, i)
            new_cap = soup.new_tag("image_caption")
            new_cap.string = '\n'.join(target_img_dict['image_caption'])
            img_tag.append(new_cap)

        if target_img_dict['image_footnote']:
            for i in target_img_dict['image_footnote']:
                text = remove_with_optional_newlines(text, i)
            new_cap = soup.new_tag("image_footnote")
            new_cap.string = '\n'.join(target_img_dict['image_footnote'])
            img_tag.append(new_cap)

        new_cap = soup.new_tag("caption")
        semantic = get_image_semantics(str(vlm_path/img_path))
        new_cap.string = semantic['description']
        img_tag.append(new_cap)

        text = remove_with_optional_newlines(text, f'![]({img_path})', '\n\n' + str(img_tag) + '\n\n')
    return text

class Ingestion:
    def __init__(self, dataset_dir,input_prefix='img',output_prefix='colqwen_ingestion',embed_model_name='vidore/colqwen2-v1.0'):
        self.dataset_dir = dataset_dir
        self.input_dir  = os.path.join(dataset_dir, input_prefix)
        self.output_dir = os.path.join(dataset_dir, output_prefix)
        self.workers = 1
        self.embed_model_name = embed_model_name
        self.reader = SimpleDirectoryReader(input_dir = self.input_dir)
        self.pipeline = IngestionPipeline(transformations=[
            VL_Embedding(model=embed_model_name,mode='text')
        ])



    def ingestion_example(self, input_files, output_file):
        # input_files: 同一文档的所有页图片路径（已排序）
        doc_id, _ = parse_doc_and_page(input_files[0])

        # 合并全文
        full_text = ""
        for f in tqdm(input_files, desc=doc_id):
            full_text += get_text(f)

        # ====== Step 1. 外部切分全文（跨页连续）======
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
        chunks = splitter.split_text(full_text)

        # 构造 nodes —— 不带页码
        nodes = [
            TextNode(text=c, metadata={"doc_id": doc_id})
            for c in chunks
        ]

        # ====== Step 3. 仅嵌入，不再切分 ======
        nodes = self.pipeline.run(nodes=nodes, num_workers=8, show_progress=False)

        nodes_json = [node.to_dict() for node in nodes]
        with open(output_file, 'w') as json_file:
            json.dump(nodes_json, json_file, indent=2, ensure_ascii=False)
        return True
    
    def ingestion_multi_session(self):
        os.makedirs(self.output_dir, exist_ok=True)
        files = os.listdir(self.input_dir)

        groups = defaultdict(list)
        for f in files:
            doc_id, page = f.rsplit("_", 1)
            page_num = int(page.split(".")[0])
            groups[doc_id].append((page_num, f))

        # 每组按页码排序
        groups = {k: [f for _, f in sorted(v)] for k, v in groups.items()}
        file_to_process = []
        for file in groups:
            input_file = [os.path.join(self.input_dir, i) for i in groups[file]]
            output_file = os.path.join(self.output_dir, file) + '.node'
            if not os.path.exists(output_file):
                file_to_process.append((input_file, output_file))
        if self.workers == 1:
            for input_file, output_file in tqdm(file_to_process):
                self.ingestion_example(input_file, output_file)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_file = {executor.submit(self.ingestion_example, input_file, output_file): (input_file, output_file) for input_file, output_file in file_to_process}
                for future in tqdm(as_completed(future_to_file), total=len(file_to_process), desc='Processing files'):
                    result_type = future.result()
    


if __name__ == '__main__':
    dataset_dir = '../search_engine_mmlongbench/corpus'
    ingestion = Ingestion(dataset_dir,input_prefix='img',output_prefix='bge_ingestion',embed_model_name='/data/user0/models/BAAI/bge-m3') # colqwen2
    ingestion.ingestion_multi_session()
