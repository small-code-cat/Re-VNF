import os
import json
# read
import argparse
import math
from tqdm import tqdm
import ast

def _to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return list(ast.literal_eval(x))
        except Exception:
            return [s.strip() for s in x.strip("[](){} ").split(",") if s.strip()]
    return [x] if x is not None else []

def recall_at_k_old(relevant_docs, retrieved_docs, k):
    recall_values = []
    for rel, ret in zip(relevant_docs, retrieved_docs):
        if k == -1:
            ret_set = set(ret)
        else:
            ret_set = set(ret[:k]) # 取前 k 个检索结果
        rel_set = set(rel)
        recall = len(rel_set & ret_set) / len(rel_set)
        recall_values.append(recall)
    return sum(recall_values) / len(recall_values)

def f0_5_at_k(relevant_docs, retrieved_docs, k):
    f0_5_values = []
    for rel, ret in zip(relevant_docs, retrieved_docs):

        # 1) 取前 k 个检索结果
        if k == -1:
            P = set(ret)
        else:
            P = set(ret[:k])

        G = set(rel)

        # 2) 计算 tp / fp / fn
        tp = len(P & G)
        fp = len(P - G)
        fn = len(G - P)

        denom = 1.25 * tp + fp + 0.25 * fn
        f0_5 =  1.25 * tp / denom

        f0_5_values.append(f0_5)

    return sum(f0_5_values) / len(f0_5_values), f0_5_values

def f1_at_k(relevant_docs, retrieved_docs, k):
    f1_values = []
    for rel, ret in zip(relevant_docs, retrieved_docs):

        # 1) 取前 k 个检索结果
        if k == -1:
            P = set(ret)
        else:
            P = set(ret[:k])

        G = set(rel)

        # 2) 计算 tp / fp / fn
        tp = len(P & G)
        fp = len(P - G)
        fn = len(G - P)

        # 3) precision / recall
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        # 4) F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1_values.append(f1)

    return sum(f1_values) / len(f1_values), f1_values


def hit_rate_at_k(relevant_docs, retrieved_docs, k):
    hit_count = 0
    for rel, ret in zip(relevant_docs, retrieved_docs):
        if k == -1:
            k = len(ret)
        if any(doc in set(ret[:k]) for doc in rel):
            hit_count += 1
    return hit_count / len(relevant_docs)

def ndcg_at_k(relevant_docs, retrieved_docs, k):
    def dcg(scores):
        return sum(score / math.log2(idx + 2) for idx, score in enumerate(scores))

    ndcg_values = []
    for rel, ret in zip(relevant_docs, retrieved_docs):
        if k == -1:
            k = len(ret)
        rel_set = set(rel)
        #  DCG@k
        scores = [1 if doc in rel_set else 0 for doc in ret[:k]]
        dcg_value = dcg(scores)
        #  IDCG@k
        ideal_scores = sorted([1] * len(rel_set) + [0] * (k - len(rel_set)), reverse=True)[:k]
        idcg_value = dcg(ideal_scores)
        ndcg = dcg_value / idcg_value if idcg_value > 0 else 0
        ndcg_values.append(ndcg)
    return sum(ndcg_values) / len(ndcg_values)

def mrr_at_k(relevant_docs, retrieved_docs, k):
    rr_values = []
    for rel, ret in zip(relevant_docs, retrieved_docs):
        if k == -1:
            k = len(ret)
        for rank, doc in enumerate(ret[:k], start=1):
            if doc in rel:
                rr_values.append(1 / rank)
                break
        else:
            rr_values.append(0)
    return sum(rr_values) / len(rr_values)

def precision_at_k(relevant_docs, retrieved_docs, k):
    precision_values = []
    for rel, ret in zip(relevant_docs, retrieved_docs):

        # 1) 取前 k 个检索结果
        if k == -1:
            P = set(ret)
        else:
            P = set(ret[:k])

        G = set(rel)

        # 2) 计算 tp / fp (Recall不需要fp，Precision不需要fn，但为了保持风格这里仅保留需要的)
        tp = len(P & G)
        fp = len(P - G)

        # 3) Precision
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        precision_values.append(precision)

    return sum(precision_values) / len(precision_values), precision_values

def recall_at_k(relevant_docs, retrieved_docs, k):
    recall_values = []
    for rel, ret in zip(relevant_docs, retrieved_docs):

        # 1) 取前 k 个检索结果
        if k == -1:
            P = set(ret)
        else:
            P = set(ret[:k])

        G = set(rel)

        # 2) 计算 tp / fn
        tp = len(P & G)
        fn = len(G - P)

        # 3) Recall
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        recall_values.append(recall)

    return sum(recall_values) / len(recall_values), recall_values

def eval_sample(example):
    retrieved_docs_list = []
    relevant_docs_list = []
    
    file_name = example['meta_info']['file_name']

def analysis(examples):
    relevant_docs_list = []
    retrieved_docs_list = []
    if examples[0].get('retrieved_images', None) is not None:
        for example in tqdm(examples):
            file_name = example['meta_info']['file_name']
            file_name = file_name.split('.')[0]
            reference_page = example['meta_info']['reference_page']
            reference_page = [str(page) for page in reference_page]
            relevant_docs_list.append([file_name + '_' + page for page in reference_page])
            retrieved_docs_list.append(_to_list(example['retrieved_images']))

        _,f1_list = f1_at_k(relevant_docs_list, retrieved_docs_list, -1)
        return [{**item, 'f1':f1_score, 'relevant_docs':relevant_docs} for item,f1_score,relevant_docs in zip(examples, f1_list, relevant_docs_list)]

def eval_search(examples):
    relevant_docs_list = []
    retrieved_docs_list = []
    results = {}
    if examples[0].get('retrieved_images',None) is not None:
        for example in tqdm(examples):
            file_name = example['meta_info']['file_name']
            file_name = file_name.split('.')[0]
            reference_page = example['meta_info']['reference_page']
            reference_page = [str(page) for page in reference_page]
            relevant_docs_list.append([file_name + '_' + page for page in reference_page])
            retrieved_docs_list.append(_to_list(example['retrieved_images']))

        for k in [-1,1,5,10]:
            # recall = recall_at_k(relevant_docs_list, retrieved_docs_list, k)
            hit_rate = hit_rate_at_k(relevant_docs_list, retrieved_docs_list, k)
            ndcg = ndcg_at_k(relevant_docs_list, retrieved_docs_list, k)
            mrr = mrr_at_k(relevant_docs_list, retrieved_docs_list, k)
            f1,_ = f1_at_k(relevant_docs_list, retrieved_docs_list, k)
            precision, _ = precision_at_k(relevant_docs_list, retrieved_docs_list, k)
            recall, _ = recall_at_k(relevant_docs_list, retrieved_docs_list, k)
            # f0_5, _ = f0_5_at_k(relevant_docs_list, retrieved_docs_list, k)
            if k == -1:
                k = 'all'
            results[f'Hit rate@{k}'] = hit_rate
            results[f'nDCG@{k}'] = ndcg
            results[f'MRR@{k}'] = mrr
            results[f'Precision@{k}'] = precision
            results[f'Recall@{k}'] = recall
            results[f'F1@{k}'] = f1
            # results[f'F0.5@{k}'] = f0_5

    if examples[0].get('eval_result',None) is not None:
        score = 0
        passing = 0
        for example in examples:
            score += example['eval_result']['score']
            passing += example['eval_result']['passing']
        results.update(dict(
            score=score / len(examples),
            passing=passing / len(examples)
        ))
    return results

def eval_search_type_wise(examples):
    examples_by_type = {}
    for example in examples:
        query_type = example['meta_info'].get('query_type', '')
        source_type = example['meta_info'].get('source_type', '')
        if '-Hop' in query_type:
            source_type = query_type.split('_')[-1]
            query_type = query_type.split('_')[0]
        if query_type != '':
            if query_type not in examples_by_type:
                examples_by_type[query_type] = []
            examples_by_type[query_type].append(example)
        if source_type != '':
            if source_type not in examples_by_type:
                examples_by_type[source_type] = []
            examples_by_type[source_type].append(example)
    results = {}
    for query_type, examples in examples_by_type.items():
        results[query_type] = eval_search(examples)
    return results
