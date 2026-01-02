import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from step_1.utils import encode_image, post_process_image
from jinja2 import Template
from tqdm.auto import tqdm
from openai import OpenAI
from PIL import Image

SEED = 42
random.seed(SEED)

img_dir = Path("/data/user0/PycharmProjects")
output_dir = Path("./page_data_v4")
output_dir.mkdir(exist_ok=True, parents=True)

key = "EMPTY"
top_k_min = 5
top_k_max = 10

client1 = OpenAI(
    # 将这里换成你在便携AI聚合API后台生成的令牌
    api_key=key,
    # 这里将官方的接口访问地址替换成便携AI聚合API的入口地址
    base_url="http://localhost:8001/v1",
)
client2 = OpenAI(
    # 将这里换成你在便携AI聚合API后台生成的令牌
    api_key=key,
    # 这里将官方的接口访问地址替换成便携AI聚合API的入口地址
    base_url="http://localhost:8002/v1",
)

with open('format_prompt/explain_noise.jinja', encoding="utf-8") as f:
    noise_template = f.read()
noise_prompt = Template(noise_template.strip())

with open('format_prompt/explain_gt.jinja', encoding="utf-8") as f:
    gt_template = f.read()
gt_prompt = Template(gt_template.strip())

def make_messages(prompt_str, image_paths):
    content_list = []
    img_idx = 0

    # 按 <image> 进行切分
    for i, text_part in enumerate(prompt_str.split("<image>")):
        # 从第二段开始，先插入一张图片
        if i != 0:
            image_path = image_paths[img_idx]
            content_list.append({
                "type": "image_url",
                "image_url": {"url": encode_image(post_process_image(Image.open(image_path).convert("RGB")))},
            })
            img_idx += 1

        # 再插入这段文本（如果非空）
        if text_part:
            content_list.append({
                "type": "text",
                "text": text_part,
            })

    return [
        {
            "role": "user",
            "content": content_list,
        }
    ]



def infer(client, messages, max_retries=5, model='Qwen-VL'):
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, timeout=100  # type: ignore
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            pass
    return None

with open('/data/user0/PycharmProjects/EasyR1-main/examples/format_prompt/noise_judge.jinja', encoding="utf-8") as f:
    prompt_template = f.read()
prompt = Template(prompt_template.strip())

with open('/data/user0/PycharmProjects/EasyR1-main/examples/format_prompt/noise_judge_no_cot.jinja', encoding="utf-8") as f:
    prompt_template = f.read()
prompt_no_cot = Template(prompt_template.strip())

def noise_judge(query, image_path):
    judge_prompt = '''You are an expert in evaluating whether a retrieved image page is relevant to a given query.

Task:
Given a user query and a retrieved page image, determine whether the image is a *noise page*.

Definition of a Noise Page:
A noise page is an image that does not meaningfully help answer the query. It may be irrelevant, unrelated in content, off-topic, or missing any visual information that could help answer the user's question. Even if the image has valid content, it is considered noise if the content does not contribute to answering the query.

Instructions:
1. Analyze the query.
2. Analyze the retrieved image.
3. Decide whether the image meaningfully supports answering the query.
4. Follow the required output format exactly.

Required Output Format (MUST follow strictly):
Explanation: <your explanation in one or two sentences>
Answer: <Yes or No>

- "Yes" means it IS a noise page.
- "No" means it is NOT a noise page.

Query:
{query}
'''
    messages = make_messages(judge_prompt.format(query=query), image_path)
    resp = infer(messages)
    answer_line = resp.split('Answer:')[-1].strip()
    if 'yes' in answer_line.lower():
        return True
    elif 'no' in answer_line.lower():
        return False
    else:
        print(f'noise judge function has wrong format extraction!')
        return None

def get_noise_think(question, image_path, positive):
    question_template = noise_prompt.render(query=question, gt_images=positive)
    messages = make_messages(question_template, positive+[image_path])
    think = infer(messages)
    return think

def get_positive_think(question, image_path):
    question_template = gt_prompt.render(query=question)
    messages = make_messages(question_template, [image_path])
    think = infer(messages)
    return think

def decide_final_label(label, model_label):
    if model_label is None:
        return label

    if label and not model_label:
        # 原始：噪声 ；模型：非噪声 → 相信模型
        return False
    if not label and model_label:
        # 原始：非噪声；模型：噪声 → 相信原始
        return False
    # 其他情况两者一致 → 直接用任意一个（都一样）
    return model_label

with open('/data/user0/PycharmProjects/MM-R5-main/format_prompt/student_reasoning.jinja', encoding="utf-8") as f:
    student_prompt_template = f.read()
student_prompt = Template(student_prompt_template.strip())
def get_student_think(query, image_path):
    question_template = student_prompt.render(query=query)
    messages = make_messages(question_template, [image_path])
    think = infer(client1, messages, model='vl-student')
    return think

with open('/data/user0/PycharmProjects/MM-R5-main/format_prompt/teacher_summary.jinja', encoding="utf-8") as f:
    teacher_prompt_template = f.read()
teacher_prompt = Template(teacher_prompt_template.strip())
def get_teacher_think(query, image_path, image_label, student_output):
    question_template = teacher_prompt.render(query=query, image_label=image_label, student_output=student_output)
    messages = make_messages(question_template, [image_path])
    think = infer(client2, messages, model='vl-teacher')
    think = think.replace("you", "I").replace("You", "I")
    sentences = think.split(".")
    sentences = [s for s in sentences if "student" not in s]
    think = ".".join(sentences).strip()
    return think

def make_data_item_student_teacher(item):
    top_k = random.randint(top_k_min, top_k_max)
    if 'label' in item:
        candidate_image = list(set(item['images'][:top_k]) - set(item['positive']))
    else:
        candidate_image = list(set(item['images'][:top_k] + item['positive']))
    noise_index = [i for i, p in enumerate(candidate_image, 1) if p not in item['positive']]
    pages = []
    question = item['problem']
    for idx, img_path in enumerate(candidate_image, 1):
        image_path = str(img_dir / img_path)
        label = idx in noise_index
        student_think = get_student_think(question, image_path)
        think = get_teacher_think(question, image_path, 'not useful' if label else 'useful', student_think)
        if think is None:
            print(f"Failed to infer for {img_path}, skipping...")
            return None
        pages.append({
            'image_path': image_path,
            'student_think': student_think,
            'think': think,
            'label': label,
        })
    random.shuffle(pages)
    data_item = {
        "messages": [
            {
                "role": "user",
                "content": prompt.render(
                    question=item['problem'],
                    images=pages,
                ),
            },
            {
                "role": "assistant",
                "content": "<think>\n"
                + "\n".join(
                    [f"Image {i}: " + p["think"] for i, p in enumerate(pages, 1)]
                )
                + "\n</think>",
            },
        ],
        "think": [p['think'] for p in pages],
        "student_think": [p['student_think'] for p in pages],
        "images": [p['image_path'] for p in pages],
        "problem": item['problem'],
        "noise_index": [i for i, p in enumerate(pages, 1) if p["label"]],
    }
    data_item["messages"][-1]["content"] += f'<answer>[{", ".join(map(str, data_item["noise_index"]))}]</answer>'
    return data_item

def make_data_item(item):
    top_k = random.randint(top_k_min, top_k_max)
    if 'label' in item:
        candidate_image = list(set(item['images'][:top_k]) - set(item['positive']))
    else:
        candidate_image = list(set(item['images'][:top_k] + item['positive']))
    noise_index = [i for i, p in enumerate(candidate_image, 1) if p not in item['positive']]
    pages = []
    question = item['problem']
    positive = [str(img_dir / i) for i in item['positive']]
    for idx, img_path in enumerate(candidate_image, 1):
        image_path = str(img_dir / img_path)
        label = idx in noise_index
        if label:
            think = get_noise_think(question, image_path, positive)
        else:
            think = get_positive_think(question, image_path)
        if think is None:
            print(f"Failed to infer for {img_path}, skipping...")
            return None
        pages.append({
            'image_path': image_path,
            'think': think,
            'label': label,
        })
    random.shuffle(pages)
    data_item = {
        "messages": [
            {
                "role": "user",
                "content": prompt.render(
                    question=item['problem'],
                    images=pages,
                ),
            },
            {
                "role": "assistant",
                "content": "<think>\n"
                + "\n".join(
                    [f"Image {i}: " + p["think"] for i, p in enumerate(pages, 1)]
                )
                + "\n</think>",
            },
        ],
        "images": [p['image_path'] for p in pages],
        "problem": item['problem'],
        "noise_index": [i for i, p in enumerate(pages, 1) if p["label"]],
    }
    data_item["messages"][-1]["content"] += f'<answer>[{", ".join(map(str, data_item["noise_index"]))}]</answer>'
    return data_item

def make_base_data_item(item):
    # 1. 筛选：Positive优先 + 去重 + 截断
    candidate_image = list(dict.fromkeys(item['positive'] + item['images']))[:10]
    noise_index = [i for i, p in enumerate(candidate_image, 1) if p not in item['positive']]
    pages = []
    for idx, img_path in enumerate(candidate_image, 1):
        image_path = str(img_dir / img_path)
        label = idx in noise_index
        pages.append({
            'image_path': image_path,
            'label': label,
        })
    random.shuffle(pages)
    data_item = {
        "messages": [
            {
                "role": "user",
                "content": prompt_no_cot.render(
                    question=item['problem'],
                    images=pages,
                ),
            },
            {
                "role": "assistant",
                "content": "",
            },
        ],
        "images": [p['image_path'] for p in pages],
        "problem": item['problem'],
        "noise_index": [i for i, p in enumerate(pages, 1) if p["label"]],
    }
    data_item["messages"][-1]["content"] += f'<answer>[{", ".join(map(str, data_item["noise_index"]))}]</answer>'
    return data_item

def main():
    src = output_dir / "sft.jsonl"
    dst = output_dir / "sft_student_teacher_test.jsonl"

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
        futures = [executor.submit(make_data_item_student_teacher, item) for item in data[:5]]

        with open(dst, "a", encoding="utf-8") as f:
            for future in tqdm(as_completed(futures), total=len(futures)):
                item = future.result()
                if item is None:
                    continue
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()

if __name__ == '__main__':
    main()