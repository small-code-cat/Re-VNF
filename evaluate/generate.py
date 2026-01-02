import json
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
from tqdm import tqdm
import argparse
import math
from evaluate.visual_rag_api import VisualRAG
from utils.common_utils import RAGMeter


def generate_item(item, visual_rag, retriever, method, args, meter):
    question = item['query']
    if 'GT' in args.method_name:
        if 'GT' == args.method_name:
            image_path = [f"../search_engine_vidoseek/corpus/img/{os.path.splitext(item['meta_info']['file_name'])[0]}_{p}.jpg" for p in item['meta_info']['reference_page']]
        else:
            GT = [
                f"../search_engine_vidoseek/corpus/img/{os.path.splitext(item['meta_info']['file_name'])[0]}_{p}.jpg"
                for p in item['meta_info']['reference_page']]
            ratio = float(args.method_name.split('_')[-1]) # 设定的噪声比例
            retrieved = retriever.search(question)

            # 1. 剔除检索结果中的 GT，保留剩下的作为噪声候选（保持原检索顺序）
            # 为了加速查找，建议先将 image_path 转为 set，但在列表推导式中直接用也可以
            noise_pool = [img for img in retrieved if img not in GT]

            # 2. 计算需要多少张噪声图片
            num_noise = math.ceil(len(GT) * ratio / (1 - ratio))

            # 确定要保留的所有图片集合 (所有GT + 选出的N个噪声)
            keep_set = set(GT + noise_pool[:num_noise])

            # 1. 按照检索结果的原顺序提取 (自动包含被检索到的GT和选定噪声)
            image_path = [img for img in retrieved if img in keep_set]

            # 2. 将未被检索到的GT追加到末尾
            # (检查 GT 是否已在 final_candidates 中，不在则追加)
            image_path += [img for img in GT if img not in image_path]
    else:
        image_path = retriever.search(question)

    stats_info = {}
    think_content = None
    before_filter = None
    if method:
        if 'filter' in args.method_name:
            (think_content, predicted_order), filter_stats = meter.measure(
                method.find_noise,
                question, image_path,
                task_type="filter",
                input_text=question,
                num_imgs=len(image_path),
                output_parser=lambda res: f"<think>{res[0]}</think><answer>{str(res[1])}</answer>"  # 提取 think 内容
            )
            stats_info['filter_stats'] = filter_stats
            before_filter = image_path
            image_path = [image_path[i] for i in range(len(image_path)) if i not in predicted_order]
        elif 'MM-R5' in args.method_name:
            (think_content, predicted_order), rerank_stats = meter.measure(
                method.rerank,
                question, image_path,
                task_type="rerank",
                input_text=question,
                num_imgs=len(image_path),
                output_parser=lambda res: f"<think>{res[0]}</think><answer>{str(res[1])}</answer>"  # 提取 think 内容
            )
            stats_info['rerank_stats'] = rerank_stats
            if 'top' in args.method_name:
                k = int(args.method_name.split('_')[-1])  # 设定的噪声比例
            image_path = [image_path[i] for i in predicted_order][:k]

    retrieved_images = [os.path.splitext(os.path.basename(i))[0] for i in image_path]
    result = {**item, 'retrieved_images': retrieved_images, **stats_info}
    if args.is_generate == 1:
        # [Wrapper] 统计 Generate 阶段 (vLLM)
        model_answer, gen_stats = meter.measure(
            visual_rag.generate,
            question, image_path,
            task_type="gen",
            input_text=question,
            num_imgs=len(image_path),  # 这里是过滤后的数量
            output_parser=lambda res: res  # 假设 visual_rag.generate 直接返回 string
        )
        result = {**result, 'model_answer': model_answer, 'gen_stats': gen_stats}
    if args.is_subset == 1:
        result = {**result, 'think_content': think_content, 'before_filter':before_filter}
    return result

def main(args):
    json_path = f'/data/user0/datasets/autumncc/ViDoSeek/{args.dataset_name}.json'
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.is_subset == 1:
        with open(os.path.join(args.save_dir, f"{args.dataset_name}_cases_qid.json"), 'r', encoding='utf-8') as f:
            uid_list = json.load(f)
        data['examples'] = [x for x in data['examples'] if x['uid'] in uid_list]

    meter = RAGMeter(model_path="/data/user0/models/Qwen/Qwen2.5-VL-7B-Instruct")
    visual_rag = VisualRAG()
    from evaluate.retriever.colqwen import ColQwenRetriever
    retriever = ColQwenRetriever(args.search_url, args.top_k)

    method = None
    if 'filter' in args.method_name:
        from evaluate.filter.noiser_vllm import QueryNoiser
        is_cot = False if 'no_cot' in args.method_name else True
        if 'drpo' in args.method_name:
            model_path = "/data/user0/PycharmProjects/EasyR1-main/checkpoints/easy_r1/qwen2_5_vl_7b_noise_judge_drpo_v2/global_step_35/actor/huggingface"
        elif 'rl' in args.method_name:
            model_path = "/data/user0/PycharmProjects/EasyR1-main/checkpoints/easy_r1/qwen2_5_vl_7b_noise_judge_grpo_sft_to_rl/global_step_170/actor/huggingface"
        elif 'sft' in args.method_name:
            model_path = "/data/user0/PycharmProjects/MM-R5-main/page_merged_qwen2_5_vl_sft_v4"
            if 'ST' in args.method_name:
                model_path = '/data/user0/PycharmProjects/MM-R5-main/paged_merged_qwen2_5_vl_sft_v5'
        elif 'base' in args.method_name:
            model_path = "/data/user0/PycharmProjects/MM-R5-main/paged_merged_qwen2_5_vl_sft_base"
        elif 'f1' in args.method_name:
            model_path = "/data/user0/PycharmProjects/EasyR1-main/checkpoints/easy_r1/qwen2_5_vl_7b_noise_judge_grpo_standard_f1/global_step_74/actor/huggingface"
        elif 'grpo' in args.method_name:
            model_path = '/data/user0/PycharmProjects/EasyR1-main/checkpoints/easy_r1/qwen2_5_vl_7b_noise_judge_grpo/global_step_74/actor/huggingface'
            if 'ST' in args.method_name:
                model_path = '/data/user0/PycharmProjects/EasyR1-main/checkpoints/easy_r1/qwen2_5_vl_7b_noise_judge_grpo_student_teacher/global_step_60/actor/huggingface'
        elif '7b' in args.method_name:
            model_path = "/data/user0/models/Qwen/Qwen2.5-VL-7B-Instruct"
        elif '32b' in args.method_name:
            model_path = "/data/user0/models/Qwen/Qwen2.5-VL-32B-Instruct"
        elif '72b' in args.method_name:
            model_path = "/data/user0/models/Qwen/Qwen2.5-VL-72B-Instruct"
        elif '12b' in args.method_name:
            model_path = "/data/user0/models/google/gemma-3-12b-it"

        print(f'{args.method_name} model_path: {model_path} is_cot: {is_cot}')
        method = QueryNoiser(model_path, is_cot)
    elif 'MM-R5' in args.method_name:
        from evaluate.reranker.rerank_vllm import QueryReranker
        method = QueryReranker("/data/user0/models/i2vec/MM-R5")

    rows = [generate_item(it, visual_rag, retriever, method, args, meter) for it in tqdm(data['examples'], desc='Generate Answer')]

    filename = (
        f"{args.dataset_name}_"
        f"{args.retriever_name}_"
        f"{args.method_name}_"
        f"{args.top_k}"
        f"{f'_{args.answer_model}' if args.is_generate == 1 else ''}"
        "_results.json"
    )

    with open(os.path.join(args.save_dir, filename), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever_name', default="ColQwen")
    parser.add_argument('--search_url', default='http://0.0.0.0:8002/search')
    parser.add_argument('--save_dir', default="demo")
    parser.add_argument('--top_k', default=10)
    parser.add_argument('--answer_model', default='Qwen-2.5-VL-7B-Instruct')
    parser.add_argument('--is_generate', default=1)
    parser.add_argument('--is_subset', default=0)

    parser.add_argument('--dataset_name', default="vidoseek") # slidevqa_refined   vidoseek
    parser.add_argument('--method_name', default="MM-R5_top_5") # MM-R5  filter_grpo naive
    args = parser.parse_args()
    main(args)