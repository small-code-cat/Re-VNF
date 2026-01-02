import os
import json
from search_engine_utils.search_engine import SearchEngine
from pathlib import Path
from datasets import load_dataset

root_dir = Path('/data/user0/datasets/MMDocIR')

def search(search_engine, dataset_dir, query):
    if isinstance(query,str):
        query = [query]
    results_batch = search_engine.search_with_score(query)
    results_batch = [
        (os.path.join(dataset_dir, "img", file), score)
        for file, score in results_batch[0]
    ]
    return results_batch

def process(domain):
    print(domain)
    dataset_dir = f'../search_engine_{domain}/corpus'
    search_engine = SearchEngine(dataset_dir, node_dir_prefix='colqwen_ingestion',
                                 embed_model_name='/data/user0/models/vidore/colqwen2-v1.0')
    data = []
    json_file = root_dir / 'MMDocIR_Train_Dataset' / f"annotations_top1_negative/{domain}_train.jsonl"
    train_data = load_dataset("json", data_files=str(json_file))["train"]
    for item in train_data:
        query = item['query'].strip()
        search_result = [(i.lstrip("../"), s) for i,s in search(search_engine, dataset_dir, query)]
        positive = [
            os.path.join(dataset_dir, 'img', f"{i['doc_name'].replace('/', '_')}_{i['page_id']}.jpg").lstrip("../") for
            i
            in item['positive_passages']]
        image_path = [i for i,_ in search_result]
        scores = [s for _, s in search_result]
        if not set(positive).issubset(set(image_path)):
            continue
        data.append({
            'problem': query,
            'domain': domain,
            'images': image_path,
            'scores': scores,
            'positive': positive
        })
    return data

def main(save_dir='rl_noise_data_construction'):
    jsonl_path = os.path.join(save_dir, 'all_page_data.jsonl')
    os.makedirs(save_dir, exist_ok=True)

    data = []
    dataset_domain = ["DUDE", "ArxivQA", 'MP-DocVQA', "SciQAG", "TAT-DQA", "Wiki-ss"]

    for domain in dataset_domain:
        domain_data = process(domain)
        data.extend(domain_data)

    # 改：写成 JSONL，每条记录一行
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    main()