from utils.qwenvl_vllm_v2 import SimpleVLLM
from utils.common_utils import parse_predicted_order

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class QueryReranker:
    """
    Universal query reranker that supports any model for image reranking
    """

    def __init__(self, model_path: str):
        """
        Initialize the reranker

        Args:
            model_path: Model path
        """
        jinja_path = 'demo/reranker/rerank.jinja'
        self.sim_vllm = SimpleVLLM(model_path, jinja_path)

    def rerank(self, query: str, images, max_retries=1):
        """
        Rerank query results

        Args:
            query: Query text
            image_list: List of image paths

        Returns:
            List[int]: Reranked index list
        """
        for attempt in range(1, max_retries + 1):
            output_text = self.sim_vllm.generate(query, images, system_prompt=SYSTEM_PROMPT)[0]
            # === 3. 尝试解析 ===
            think_content, predicted_order = parse_predicted_order(output_text, images)

            if predicted_order is not None:
                return think_content, predicted_order  # 成功解析，直接返回

            print(f"Retry {attempt}/{max_retries} failed. Retrying...")
        return None, []
