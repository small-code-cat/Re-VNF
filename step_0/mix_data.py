import json
import os
import random
SEED = 42
random.seed(SEED)

def main(save_dir='rl_noise_data_construction'):
    all_data = []

    # gt page > 1
    gt_n_data = []
    with open(os.path.join(save_dir, 'page_data_MMDocRAG.jsonl'), "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            gt_n_data.append(item)
    all_data.extend(gt_n_data)

    # gt page == 1
    gt_1_data = []
    gt_1_query = []
    with open(os.path.join(save_dir, 'all_page_data.jsonl'), "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item['problem'] not in gt_1_query:
                gt_1_data.append(item)
                gt_1_query.append(item['problem'])
    sample = random.sample(gt_1_data, 8000)
    sample_query = [i['problem'] for i in sample]
    remaining = [item for item in gt_1_data if item["problem"] not in sample_query]
    all_data.extend(sample)

    # gt page == 0
    # sample = random.sample(remaining, 2000)
    # all_data.extend([{**i, 'label': 0} for i in sample])

    # 改：写成 JSONL，每条记录一行
    jsonl_path = os.path.join(save_dir, 'page_data_mix_base.jsonl')
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    main()