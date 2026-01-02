from utils.qwenvl_vllm_api import get_client, infer
from utils.common_utils import answer_prompt

class VisualRAG:
    def __init__(self):
        self.client = get_client()

    def generate(self, query, image_path):
        answer = infer(self.client, answer_prompt, f'Question:\n{query}\n', image_path)
        return answer