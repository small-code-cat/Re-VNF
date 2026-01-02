import json
import random
from pathlib import Path
from jinja2 import Template
from concurrent.futures import ThreadPoolExecutor, as_completed

import re
from tqdm.auto import tqdm
from openai import OpenAI

SEED = 42
random.seed(SEED)

output_dir = Path("./page_data_v4")
output_dir.mkdir(exist_ok=True, parents=True)

base_url = "http://localhost:8003/v1"
key = "EMPTY"
model = "Qwen"

client = OpenAI(
    # 将这里换成你在便携AI聚合API后台生成的令牌
    api_key=key,
    # 这里将官方的接口访问地址替换成便携AI聚合API的入口地址
    base_url=base_url,
)

with open('format_prompt/think_summary.jinja', encoding="utf-8") as f:
    prompt_template = f.read()
think_summary_prompt = Template(prompt_template.strip())

with open('format_prompt/extract_imageid.jinja', encoding="utf-8") as f:
    prompt_template = f.read()
extract_id_prompt = Template(prompt_template.strip())

def infer(messages, max_retries=5):
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, timeout=100  # type: ignore
            )
            content = response.choices[0].message.content
            return content
        except:
            pass
    return None

def get_think_summary(question, noise_image, gt_image, think):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": think_summary_prompt.render(
                        question=question, noise_image=noise_image, gt_image=gt_image, think=think
                    ),
                },
            ],
        },
    ]
    response = infer(messages)
    return response

def get_irrelevant_id(question, think):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": extract_id_prompt.render(
                        question=question, think=think
                    ),
                },
            ],
        },
    ]
    response = infer(messages)
    match = re.search(r'\[.*?\]', response, re.S)
    ids = json.loads(match.group()) if match else []
    return ids

def make_data_item(item):
    content = item["messages"][-1]["content"]
    idx = content.find("<answer>")
    think = content[:idx][7:-8].strip()
    # answer = content[idx:].strip()
    # gt_image = sorted(set(range(1, len(item['images']) + 1))-set(item['noise_index']))
    # think_sum = get_think_summary(
    #     question=item["problem"],
    #     noise_image=", ".join(map(str, item["noise_index"])),
    #     gt_image=", ".join(map(str, gt_image)),
    #     think=think,
    # )
    # if think_sum is None:
    #     print(f"Failed to summarize for {item['problem']}")
    #     return None
    irrelevant_id = get_irrelevant_id(item["problem"], think)
    if set(item["noise_index"]) != set(irrelevant_id):
        print(f'think content is consistent with noise index')
        return None
    # content_new = "<think>" + think_sum + "</think>" + answer
    # item["messages"][-1]["content"] = content_new
    return item

def make_base_data_item(item):
    content = item["messages"][-1]["content"]
    idx = content.find("<answer>")
    think = content[:idx][7:-8].strip()
    answer = content[idx:].strip()
    gt_image = sorted(set(range(1, len(item['images']) + 1))-set(item['noise_index']))
    think_sum = get_think_summary(
        question=item["problem"],
        noise_image='',
        gt_image=", ".join(map(str, gt_image)),
        think=think,
    )
    if think_sum is None:
        print(f"Failed to summarize for {item['problem']}")
        return None
    content_new = "<think>" + think_sum + "</think>" + answer
    item["messages"][-1]["content"] = content_new
    return item

def main():
    src = output_dir / "sft_student_teacher.jsonl"
    dst = output_dir / "sft_student_teacher_2.jsonl"

    # 已处理过的 query
    done = set()
    if dst.exists():
        with open(dst, "r", encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["problem"])

    # 加载并过滤原始数据
    data = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["problem"] not in done:
                data.append(item)

    random.shuffle(data)

    # 异步处理 + 实时写入
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(make_data_item, item) for item in data]

        with open(dst, "a", encoding="utf-8") as f:
            for future in tqdm(as_completed(futures), total=len(futures)):
                item = future.result()
                if item is None:
                    continue
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()

if __name__ == '__main__':
    main()