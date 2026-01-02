# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any
from mathruler.grader import extract_boxed_content, grade_answer

def format_reward(response: str, n: int) -> float:
    think_ok = re.search(r"<think>.*?</think>", response, flags=re.S) is not None
    ans_match = re.search(r"<answer>\s*\[([^\]]*)\]\s*</answer>", response, flags=re.S)
    if not (think_ok and ans_match):
        return 0.0

    raw = ans_match.group(1).strip()

    # 空列表情况：[]
    if raw == "":
        return 1.0

    tokens = [t.strip() for t in raw.split(",") if t.strip()]

    # 如果 tokens 非空，但里面出现非数字 → 格式错误
    if any(not t.isdigit() for t in tokens):
        return 0.0

    # 此时 tokens 全是数字
    idx_list = []
    for t in tokens:
        v = int(t)
        if v not in idx_list:
            idx_list.append(v)

    if len(idx_list) == 0:
        # 理论上不出现，因为 raw != "" 且全为数字
        return 0.0

    # 计算范围奖励
    valid_cnt = sum(1 for v in idx_list if 1 <= v <= n)
    return valid_cnt / len(idx_list)

def accuracy_reward(response: str, ground_truth: list[int]) -> float:
    m = re.search(r"<answer>\s*\[([^\]]*)\]\s*</answer>", response, flags=re.S)
    if not m:
        return 0.0

    try:
        pred = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
    except Exception:
        return 0.0

    P, G = set(pred), set(ground_truth)

    tp = len(P & G)
    fp = len(P - G)
    fn = len(G - P)

    denom = 2 * tp + 2 * fp + fn
    return 2 * tp / denom


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.0) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        format_score = format_reward(response, len(reward_input["images"]))
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
