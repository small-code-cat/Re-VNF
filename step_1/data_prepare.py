import random
from collections import defaultdict
import json
import os

# data: list[dict]，元素形如你给的那个 {"query": ..., "domain": ..., "positive": [...], "negative": [...]}

random.seed(42)
sft_ratio = 0.7

json_path = ''
all_data = []
with open(json_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        all_data.append(item)
    
OUTPUT_DIR = "page_data_base"
os.makedirs(OUTPUT_DIR, exist_ok=True)

by_domain = defaultdict(list)
for ex in all_data:
    by_domain[ex["domain"]].append(ex)

sft_rows, rl_rows = [], []

for domain, examples in by_domain.items():
    random.shuffle(examples)
    n_sft = int(len(examples) * sft_ratio)
    sft_ex = examples[:n_sft]
    rl_ex = examples[n_sft:]
    sft_rows.extend(sft_ex)
    rl_rows.extend(rl_ex)

# 保存为 parquet
sft_path = os.path.join(OUTPUT_DIR, "sft.jsonl")
with open(sft_path, "w", encoding="utf-8") as f:
    for item in sft_rows:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
rl_path = os.path.join(OUTPUT_DIR, "rl.jsonl")
with open(rl_path, "w", encoding="utf-8") as f:
    for item in rl_rows:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")