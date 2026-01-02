import random
from collections import defaultdict
import pandas as pd
import json
import os

# data: list[dict]，元素形如你给的那个 {"query": ..., "domain": ..., "positive": [...], "negative": [...]}

random.seed(42)
train_ratio = 0.8

jsonl_path = '/data/user0/PycharmProjects/MM-R5-main/page_data_v4/sft_to_rl.jsonl'
data = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        data.append(item)

OUTPUT_DIR = "layout_noise_judge_output" if 'layout' in jsonl_path else "page_noise_judge_output/sft_to_rl"
os.makedirs(OUTPUT_DIR, exist_ok=True)

by_domain = defaultdict(list)
for ex in data:
    by_domain[ex["domain"]].append(ex)

train_rows, test_rows = [], []

for domain, examples in by_domain.items():
    random.shuffle(examples)
    n_train = int(len(examples) * train_ratio)
    train_ex = examples[:n_train]
    test_ex = examples[n_train:]

    for split_ex, collector in ((train_ex, train_rows), (test_ex, test_rows)):
        for ex in split_ex:
            collector.append({
                "problem": ex["problem"],
                "images": ex["images"],  # 直接用新的 images 列表
                "answer": ex["answer"],
                'label': ex['label']
            })

train_df = pd.DataFrame(train_rows)
test_df = pd.DataFrame(test_rows)

# 保存为 parquet
train_path = os.path.join(OUTPUT_DIR, "train.parquet")
test_path = os.path.join(OUTPUT_DIR, "test.parquet")

train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)

print(f"train size: {len(train_df)}, saved to: {train_path}")
print(f"test size: {len(test_df)}, saved to: {test_path}")