import numpy as np
from openai import OpenAI
import statistics
import os
import argparse
import json
from tqdm import tqdm
import ast
from concurrent.futures import ThreadPoolExecutor
from llms.evaluator import Evaluator
from utils.overall_evaluator import eval_search,eval_search_type_wise

evaluator = Evaluator()

def _to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return list(ast.literal_eval(x))
        except Exception:
            return [s.strip() for s in x.strip("[](){} ").split(",") if s.strip()]
    return [x] if x is not None else []

def eval_item(item):
    query = item['query']
    gt_answer = item['reference_answer']
    if 'model_answer' not in item:
        return item
    model_answer = item['model_answer']

    eval_result = evaluator.evaluate(query, gt_answer, model_answer)

    return {**item, 'eval_result': eval_result}

def print_eval_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    results = eval_search(rows)
    print('=============OVERALL EVALUATION SUMMARY=============')
    print_dict(results)

def print_dict(d):
    print(json.dumps(d, indent=2, ensure_ascii=False))


def print_metrics(rows):
    """
    输入: rows (包含所有样本结果的列表)
    输出: 打印平均 Latency 和 Total Tokens
    返回: (avg_latency, avg_tokens) 元组
    """
    if not rows:
        print("⚠️ No data found in rows.")
        return 0, 0

    count = len(rows)

    # --- 1. 提取各阶段 Latency (ms) ---
    # 使用 .get() 提供默认值 0，防止某些方法没有该阶段
    lat_filter = [r.get('filter_stats', {}).get('latency', 0) for r in rows]
    lat_rerank = [r.get('rerank_stats', {}).get('latency', 0) for r in rows]
    lat_gen = [r.get('gen_stats', {}).get('latency', 0) for r in rows]

    # 计算总平均时间 (End-to-End Latency)
    avg_lat_filter = sum(lat_filter) / count
    avg_lat_rerank = sum(lat_rerank) / count
    avg_lat_gen = sum(lat_gen) / count
    avg_total_latency = avg_lat_filter + avg_lat_rerank + avg_lat_gen

    # --- 2. 提取各阶段 Total Tokens (Input + Output) ---
    # 这里的 total_tokens 包含了 (Query + Image + SystemPrompt) + (Reasoning + Answer)
    tok_filter = [r.get('filter_stats', {}).get('total_tokens', 0) for r in rows]
    tok_rerank = [r.get('rerank_stats', {}).get('total_tokens', 0) for r in rows]
    tok_gen = [r.get('gen_stats', {}).get('total_tokens', 0) for r in rows]

    # 计算总平均 Token
    avg_tok_filter = sum(tok_filter) / count
    avg_tok_rerank = sum(tok_rerank) / count
    avg_tok_gen = sum(tok_gen) / count
    avg_total_tokens = avg_tok_filter + avg_tok_rerank + avg_tok_gen

    # --- 3. 打印漂亮的报告 ---
    print("\n" + "=" * 50)
    print(f"📊 Overall Performance Report (Samples: {count})")
    print("=" * 50)

    print(f"⏱️  Avg Latency (Total):     {avg_total_latency:.2f} ms")
    if avg_lat_filter > 0:
        print(f"    ├─ Filter Phase:         {avg_lat_filter:.2f} ms")
    if avg_lat_rerank > 0:
        print(f"    ├─ Rerank Phase:         {avg_lat_rerank:.2f} ms")
    print(f"    └─ Generation Phase:     {avg_lat_gen:.2f} ms")

    print("-" * 50)

    print(f"🪙  Avg Total Tokens:        {avg_total_tokens:.1f}")
    if avg_tok_filter > 0:
        print(f"    ├─ Filter (In+Out):      {avg_tok_filter:.1f}")
    if avg_tok_gen > 0:
        print(f"    └─ Generation (In+Out):  {avg_tok_gen:.1f}")

    print("=" * 50 + "\n")

    return avg_total_latency, avg_total_tokens

def eval_all(items):
    with ThreadPoolExecutor(max_workers=64) as ex:
        rows = list(tqdm(ex.map(eval_item, items), total=len(items)))

    results = eval_search(rows)
    print('=============OVERALL EVALUATION SUMMARY=============')
    print_dict(results)

    print_metrics(rows)

    # results = eval_search_type_wise(rows)
    # print('=============TYPE EVALUATION SUMMARY=============')
    # print_dict(results)
    return rows


# ===== 示例 =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever_name', default="ColQwen")
    parser.add_argument('--save_dir', default="demo")
    parser.add_argument('--top_k', default=10)
    parser.add_argument('--answer_model', default='Qwen-2.5-VL-7B-Instruct')
    parser.add_argument('--is_generate', default=1)

    parser.add_argument('--dataset_name', default="vidoseek")  # slidevqa_refined   vidoseek
    parser.add_argument('--method_name', default="filter_7b_no_cot")  # MM-R5  filter_grpo naive
    args = parser.parse_args()
    json_name = (
        f"{args.dataset_name}_"
        f"{args.retriever_name}_"
        f"{args.method_name}_"
        f"{args.top_k}"
        f"{f'_{args.answer_model}' if args.is_generate == 1 else ''}"
    )
    with open(f'/data/user0/PycharmProjects/VRAG-main/demo/{json_name}_results.json', "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = eval_all(data)
    with open(os.path.join(args.save_dir, f"{json_name}_eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)
    # print_eval_json(os.path.join(args.save_dir, f"{json_name}_eval_results.json"))