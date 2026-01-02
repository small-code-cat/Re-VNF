import json
import os
from collections import Counter
from search_engine_utils.search_engine import SearchEngine
from tqdm import tqdm

root_dir = '/data/user0/datasets/MMDocIR/MMDocRAG'
img_dir = 'search_engine_MMDocRAG/corpus/img'

def search(search_engine, dataset_dir, query):
    if isinstance(query,str):
        query = [query]
    results_batch = search_engine.search_with_score(query, top_k=5000)
    results_batch = [
        (os.path.join(dataset_dir, "img", file), score)
        for file, score in results_batch[0]
    ]
    return results_batch

# 构造 page → path 的小函数
def make_path(doc, pid):
    return f"{img_dir}/{doc}_{pid + 1}.jpg"

def process_item(item):
    doc_name = item['doc_name']
    # 所有文本和图片 quotes 合在一起，方便处理
    all_quotes = item['text_quotes'] + item['img_quotes']

    # gold quotes → 找对应的 quote → 取 page_id（去重）
    positive = sorted({
        make_path(doc_name, q['page_id'])
        for q in all_quotes
        if q['quote_id'] in item['gold_quotes']
    })

    return {
        'problem': item['question'],
        'domain': item['domain'],
        'positive': positive
    }

def process_final_data(processed_data):
    domain = 'MMDocRAG'
    dataset_dir = f'../search_engine_{domain}/corpus'
    search_engine = SearchEngine(dataset_dir, node_dir_prefix='colqwen_ingestion',
                                 embed_model_name='/data/user0/models/vidore/colqwen2-v1.0')
    data = []
    for item in tqdm(processed_data):
        query = item['problem'].strip()
        positive = item['positive']
        domain = item['domain']
        search_result = [(i.lstrip("../"), s) for i, s in search(search_engine, dataset_dir, query)]
        img_score = dict(search_result)
        if not set(positive).issubset(set(dict(search_result))):
            print(f"Skipping {query}")
            continue
        image_path = list(set([i for i, _ in search_result[:20]]+positive))
        scores = [img_score[i] for i in image_path]
        data.append({
            'problem': query,
            'domain': domain,
            'images': image_path,
            'scores': scores,
            'positive': positive
        })
    return data

def main(save_dir='rl_noise_data_construction'):
    jsonl_path = os.path.join(save_dir, 'page_data_MMDocRAG.jsonl')
    data = []
    with open(os.path.join(root_dir, 'dev_20.jsonl'), "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    with open(os.path.join(root_dir, 'evaluation_20.jsonl'), "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    all_page_ids = {q['page_id'] for item in data for q in (item['text_quotes'] + item['img_quotes'])}
    print("Min page_id =", min(all_page_ids))

    processed_data = [process_item(item) for item in data]
    processed_data = [p for p in processed_data if len(p['positive'])>1]

    processed_data = process_final_data(processed_data)

    print("总样本数:", len(data))
    print("positive>1 的样本数:", len(processed_data))

    total = len(processed_data)
    pos_len_dist = Counter(len(p['positive']) for p in processed_data)

    print("positive 个数分布（含比例）：")
    for k in sorted(pos_len_dist):
        print(f"{k}: {pos_len_dist[k]} ({pos_len_dist[k] / total:.2%})")

    # 改：写成 JSONL，每条记录一行
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    main()