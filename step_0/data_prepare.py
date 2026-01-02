import os
from pathlib import Path
from step_0.page_level_data_build import rag_query_judge
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor  # 新增
from datasets import load_dataset

root_dir = Path('/data/user0/datasets/MMDocIR')


def process_domain(domain):
    json_file = root_dir / 'MMDocIR_Train_Dataset' / f"annotations_top1_negative/{domain}_train.jsonl"
    train_data = load_dataset("json", data_files=str(json_file))["train"]
    train_data = train_data.shuffle()

    # 单个样本的处理逻辑
    def _process_item(item):
        query = item['query'].strip()
        is_rag_query = rag_query_judge(query)
        if is_rag_query is not None and is_rag_query:
            return {**item, "domain": domain}
        return None

    data = []
    # 对每个 item 开线程处理，并用 tqdm 显示当前 domain 的进度
    with ThreadPoolExecutor(max_workers=64) as executor:
        for res in tqdm(
            executor.map(_process_item, train_data),
            total=len(train_data),
            desc=f"{domain}"
        ):
            if res is not None:
                data.append(res)

    return data


def page_main(save_dir='rl_noise_data_construction'):
    json_path = os.path.join(save_dir, 'page_data.json')
    os.makedirs(save_dir, exist_ok=True)

    data = []
    dataset_domain = ["ArxivQA", "DUDE", 'MP-DocVQA', "SciQAG", "TAT-DQA", "Wiki-ss"]

    # 这里不再对 domain 本身开多线程，而是 domain 内按 item 并行，避免线程过多
    for domain in dataset_domain:
        domain_data = process_domain(domain)
        data.extend(domain_data)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def process():
    query_list, query_indices = get_queries(
        os.path.join(root_dir, 'MMDocIR_Evaluation_Dataset', "MMDocIR_annotations.jsonl")
    )
    data = []

    # 单个 query_index 的处理逻辑
    def _process_one(item):
        query_id, start_pid, end_pid, layout_mapping, domain, doc_name = item
        query = query_list[query_id]
        is_rag_query = rag_query_judge(query)
        if is_rag_query is not None and is_rag_query:
            return {
                'query': query,
                'start_pid': start_pid,
                'end_pid': end_pid,
                'layout_mapping': layout_mapping,
                'domain': domain,
                'doc_name': doc_name
            }
        return None

    # 用线程池 + tqdm 处理所有 query_indices
    with ThreadPoolExecutor(max_workers=64) as executor:
        for res in tqdm(
            executor.map(_process_one, query_indices),
            total=len(query_indices),
            desc="layout"
        ):
            if res is not None:
                data.append(res)

    return data

def filter_rag_queries(save_dir='rl_noise_data_construction'):
    all_data = []
    with open("/data/user0/PycharmProjects/VRAG-main/rl_noise_data_construction/all_page_data.jsonl", "r",
              encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            all_data.append(item)

    with open(os.path.join(save_dir, 'page_data.json'), "r", encoding="utf-8") as f:
        page_data = json.load(f)
        page_queries = {item["query"] for item in page_data}

    filtered = [item for item in all_data if item["problem"] in page_queries]

    # 改：写成 JSONL，每条记录一行
    jsonl_path = os.path.join(save_dir, 'all_page_data_rag_query.jsonl')
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in filtered:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    filter_rag_queries()
