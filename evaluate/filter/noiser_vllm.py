from utils.qwenvl_vllm_v2 import SimpleVLLM
from utils.common_utils import parse_predicted_order

class QueryNoiser:
    """
    Universal query reranker that supports any model for image reranking
    """

    def __init__(self, model_path: str, is_cot=True):
        """
        Initialize the reranker

        Args:
            model_path: Model path
        """
        if is_cot:
            jinja_path = '/data/user0/PycharmProjects/EasyR1-main/examples/format_prompt/noise_judge.jinja'
        else:
            jinja_path = '/data/user0/PycharmProjects/EasyR1-main/examples/format_prompt/noise_judge_no_cot.jinja'
        self.sim_vllm = SimpleVLLM(model_path, jinja_path)

    def find_noise(self, query: str, images, max_retries=1):
        """
        Rerank query results

        Args:
            query: Query text
            image_list: List of image paths

        Returns:
            List[int]: Reranked index list
        """
        for attempt in range(1, max_retries + 1):
            output_text = self.sim_vllm.generate(query, images)[0]
            # === 3. 尝试解析 ===
            think_content, predicted_order = parse_predicted_order(output_text, images)

            if predicted_order is not None:
                return think_content, predicted_order  # 成功解析，直接返回

            print(f"Retry {attempt}/{max_retries} failed. Retrying...")
        return None, []
