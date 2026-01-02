import json
from pathlib import Path
import random
SEED = 42
random.seed(SEED)
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from step_1.my_gpt4o_cot import noise_judge, decide_final_label
import json


output_dir = Path("./page_data_v4")
img_dir = Path("/data/user0/PycharmProjects")
top_k_min = 5
top_k_max = 10

def make_data_item(item):
    pages = []
    for idx, image_path in enumerate(item['images'], 1):
        abs_img_path = str(img_dir / image_path)
        label = idx in item['answer']
        llm_label = noise_judge(item['problem'], abs_img_path)
        final_label = decide_final_label(label, llm_label)
        pages.append({
            'image_path': image_path,
            'label': final_label,
        })
    random.shuffle(pages)
    return {
        'problem': item['problem'],
        'domain': item['domain'],
        'images': [p['image_path'] for p in pages],
        'answer': [i for i, p in enumerate(pages, 1) if p["label"]]
    }

def label_query_noise(positive_scores, candidate_scores,
                      close_ratio=0.9,
                      high_p=0.5,
                      mid_p=0.25):
    # 用所有 GT 的平均分作为基准（兼容 1 个和多个 GT）
    baseline = sum(positive_scores) / len(positive_scores)

    # 每个候选相对 GT 基准的比例
    ratios = [c / baseline for c in candidate_scores]

    # “接近 GT 的候选”的占比
    p_close = sum(r >= close_ratio for r in ratios) / len(ratios)

    # 按占比打 label
    if p_close >= high_p:
        return "noise_high"   # 很多候选都接近 GT → 噪声多
    elif p_close >= mid_p:
        return "noise_mid"    # 一部分候选接近 GT → 噪声适中
    else:
        return "noise_low"    # 只有少数候选接近 GT → 噪声少

def make_data_item_2(item):
    problem = item['problem']
    candidate_image = list(set(item['images'][:top_k] + item['positive']))
    noise_index = [i for i, p in enumerate(candidate_image, 1) if p not in item['positive']]
    img_score = dict(zip(item['images'], item['scores']))
    positive_score = [img_score[i] for i in set(item['positive'])]
    candidate_img_score = [img_score[i] for i in (set(candidate_image) - set(item['positive']))]
    label = label_query_noise(positive_score, candidate_img_score)
    return {
        'problem': problem,
        'domain': item['domain'],
        'images': candidate_image,
        'answer': noise_index,
        'label': label
    }

def make_data_item_3(item):
    problem = item['problem']
    top_k = random.randint(top_k_min, top_k_max)
    if 'label' in item:
        candidate_image = list(set(item['images'][:top_k]) - set(item['positive']))
    else:
        candidate_image = list(set(item['images'][:top_k] + item['positive']))
    noise_index = [i for i, p in enumerate(candidate_image, 1) if p not in item['positive']]
    gt_page_num = len(candidate_image) - len(noise_index)
    if gt_page_num == 0:
        label = 'zero'
    elif gt_page_num == 1:
        label = 'one'
    elif gt_page_num > 1:
        label = 'multi'
    return {
        'problem': problem,
        'domain': item['domain'],
        'images': candidate_image,
        'answer': noise_index,
        'label': label
    }

def main():
    data = []
    with open(output_dir / 'sft.jsonl', "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    random.shuffle(data)
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for item in data:
            futures.append(executor.submit(make_data_item_3, item))

        final_data = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            data_item = future.result()
            if data_item is None:
                continue
            final_data.append(data_item)
        with open(output_dir / "sft_to_rl.jsonl", "w", encoding="utf-8") as f:
            for item in final_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    main()