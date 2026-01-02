from pathlib import Path
import pandas as pd
from PIL import Image
import json
import io
import random
SEED = 42
random.seed(SEED)
from collections import defaultdict
import os
from search_engine_utils.search_engine import SearchEngine
from utils.qwenvl_vllm_api import get_client, infer

root_dir = Path('/data/user0/datasets/MMDocIR/MMDocIR_Train_Dataset')
client = get_client(base_url="http://localhost:8003/v1")
vl_client = get_client(base_url="http://localhost:8001/v1")
N = 2000

def format_query(query: str, prefix: str = '') -> str:
    return f'{prefix} {query.strip()}'.strip()

def format_passage(text: str, title: str = '', prefix: str = '') -> str:
    return f'{prefix} {title.strip()} {text.strip()}'.strip()

def _get_image(doc_name, page_id, page_image_df):
    item_row = page_image_df[
        (page_image_df['file_name'] == doc_name) & (page_image_df['page'] == page_id)]
    if len(item_row) == 1:
        img_bytes, page_size, page_layouts = item_row["image"].iloc[0], item_row["page_size"].iloc[0], \
            item_row["layouts"].iloc[0]
        image = Image.open(io.BytesIO(img_bytes))
        return {"image": image, "page_size": page_size, "page_layouts": page_layouts, "file_name": doc_name,
                "page_id": page_id}
    else:
        raise ValueError(f"Document {doc_name} does not have page {page_id}! Please check your data")

def search(search_engine, dataset_dir, query, top_k=5):
    if isinstance(query,str):
        query = [query]
    results_batch = search_engine.batch_search(query)
    results_batch = [[dict(idx=idx, image_file=(
        os.path.join(f'{dataset_dir}/img', file) if os.path.isfile(os.path.join(f'{dataset_dir}/img', file)) else file))
                      for idx, file in enumerate(query_results)] for query_results in results_batch]
    image_path_list = [result['image_file'] for result in results_batch[0]]
    return image_path_list[:top_k]

def rag_query_judge(query):
    prompt = '''You are a classifier that determines whether a user question is a RAG-style (Retrieval-Augmented Generation) query.
    
Definition of RAG-style Query ("Yes"):
A query is RAG-style if answering it would naturally require retrieving information from external documents beyond the immediate local context. Typical cases include general factual knowledge, definitions, background theory, details not shown in the current passage/figure, or any question that remains meaningful even when no local context is provided.

Definition of Non-RAG Query ("No"):
A query is NOT RAG-style if it can be answered solely based on the local context the user is currently examining (e.g., a page, a paragraph, a figure, a table). These questions typically ask for interpretation of elements inside the provided materials, such as columns, curves, labels, symbols, or numbers.

Key Distinction:
- If the question is about interpreting something in the given figure/table/passage, classify as No.
- If the question relies on external background knowledge unrelated to a specific provided context, classify as Yes.

Output Format (MUST follow strictly):
Explanation: <one or two sentences>
Answer: <Yes or No>

User Question:
{query}
'''
    resp = infer(client, None, prompt.format(query=query), None, model='Qwen')
    answer_line = resp.split('Answer:')[-1].strip()
    if 'yes' in answer_line.lower():
        return True
    elif 'no' in answer_line.lower():
        return False
    else:
        print(f'rag query judge function has wrong format extraction!')
        return None

def process_page(data_list):
    groups = defaultdict(list)
    for item in data_list:
        groups[item['domain']].append(item)

    data = []
    for domain, items in groups.items():
        print(domain)
        dataset_dir = f'../search_engine_{domain}/corpus'
        search_engine = SearchEngine(dataset_dir, node_dir_prefix='colqwen_ingestion',
                                     embed_model_name='/data/user0/models/vidore/colqwen2-v1.0')
        sampled = []
        sampled.extend(random.sample(items, min(N, len(items))))
        for item in sampled:
            query = item['query'].strip()
            image_path = [i.lstrip("../") for i in search(search_engine, dataset_dir, query)]
            positive = [os.path.join(dataset_dir, 'img', f"{i['doc_name'].replace('/', '_')}_{i['page_id']}.jpg").lstrip("../") for i
                        in item['positive_passages']]
            # 1. 过滤 positive，移除已存在的
            positive_filtered = [x for x in positive if x not in image_path]
            # 2. 合并到 image_path
            merged = image_path + positive_filtered
            # 3. 得到 negative 的索引（合并后 merged 中不属于 original positive 的元素）
            negative_indices = [i for i, x in enumerate(merged, 1) if x not in positive]
            data.append({
                'problem': query,
                'domain': domain,
                'images': merged,
                'answer': negative_indices
            })
    return data

def process_layout(data_list):
    groups = defaultdict(list)
    for item in data_list:
        groups[item['domain']].append(item)

    dataset_dir = f'../search_engine_MMDocIR_eval/corpus'
    search_engine = SearchEngine(dataset_dir, node_dir_prefix='colqwen_ingestion',
                                 embed_model_name='/data/user0/models/vidore/colqwen2-v1.0')
    data = []
    for domain, items in groups.items():
        print(domain)
        sampled = []
        sampled.extend(random.sample(items, min(N, len(items))))
        for item in sampled:
            query = item['query'].strip()
            image_path = [i.lstrip("../") for i in search(search_engine, dataset_dir, query)]
            positive_passages = list(set([l['page'] for l in item['layout_mapping']]))
            positive = [
                os.path.join(dataset_dir, 'img', f"{item['doc_name'].replace('/', '_')}_{i}.jpg").lstrip("../")
                for i
                in positive_passages]
            # 1. 过滤 positive，移除已存在的
            positive_filtered = [x for x in positive if x not in image_path]
            # 2. 合并到 image_path
            merged = image_path + positive_filtered
            # 3. 得到 negative 的索引（合并后 merged 中不属于 original positive 的元素）
            negative_indices = [i for i, x in enumerate(merged, 1) if x not in positive]
            data.append({
                'problem': query,
                'domain': domain,
                'images': merged,
                'answer': negative_indices
            })
    return data

def build_pages(domain):
    print(domain)
    save_path = f'../search_engine_{domain}/corpus/img'
    os.makedirs(save_path, exist_ok=True)
    parquet_file = root_dir / f"parquet/{domain}_filter.parquet"
    df = pd.read_parquet(str(parquet_file))
    for index, row in df.iterrows():
        file_name = row['file_name'].replace("/", "_")
        page_id = int(row['page'])
        image = Image.open(io.BytesIO(row["image"]))
        img_path = os.path.join(save_path, f'{file_name}_{page_id}.jpg')
        if os.path.exists(img_path):
            print(f'The image {img_path} already exists!')
        image.save(img_path, 'JPEG')

def main(save_dir='rl_noise_data_construction'):
    data = []
    json_path = os.path.join(save_dir, 'page_data_1.json')
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    with open(os.path.join(save_dir, 'page_data.json'), "r", encoding="utf-8") as f:
        page_data = json.load(f)
    data.extend(process_page(page_data))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()