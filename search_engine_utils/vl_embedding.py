import asyncio
from typing import Any, List, Optional, Union
import torch
from PIL import Image

from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.base.embeddings.base import Embedding

from colpali_engine.models import ColQwen2, ColQwen2Processor, ColPali, ColPaliProcessor


def weighted_mean_pooling(hidden, attention_mask):
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps

device = 'cuda:0'

def fix(img):
    w, h = img.size
    return img.resize((max(w, 28), max(h, 28)))

def get_embedding(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    bs = last_hidden_state.shape[0]
    reps = last_hidden_state[torch.arange(bs, device=last_hidden_state.device), sequence_lengths]
    reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps

class VL_Embedding(MultiModalEmbedding):

    model: str = Field(description="The Multi-model to use.")

    device: str = Field(default=device, description="The Multi-model to use.")

    api_key: Optional[str] = Field(
        default=None,
        description="The API key.",
    )
    dimensions: Optional[int] = Field(
        default=1024,
        description=(
            "The number of dimensions the resulting output embeddings should have. "
            "Only supported in embedding-3 and later models. embedding-2 is fixed at 1024."
        ),
    )
    timeout: Optional[float] = Field(
        default=None,
        description="The timeout.",
    )

    mode: str = Field(
        default='text',
        description="The mode of the model, either 'text' or 'image'."
    )
    show_progress: bool = Field(
        default=False,
        description="Whether to show progress bars.",
    )
    
    embed_model: Union[ColQwen2, AutoModel, None] = Field(
        default=None
    )
    processor: Optional[ColQwen2Processor] = Field(
        default=None
    )
    tokenizer: Optional[AutoTokenizer] = Field(
        default=None
    )
    
    
    def __init__(
        self,
        model: str = "vidore/colqwen2-v1.0",
        dimensions: Optional[int] = 1024,
        timeout: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        mode: str = 'text',
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            dimensions=dimensions,
            timeout=timeout,
            callback_manager=callback_manager,
            **kwargs,
        )
        
        self.mode = mode
        self.device = device
        
        if 'openbmb' in model:
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            self.embed_model = AutoModel.from_pretrained(model,
             torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device).cuda().eval()
            # self.embed_model.eval()
        elif 'vidore' in model and 'qwen' in model:
            self.embed_model = ColQwen2.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map=device,  # or "mps" if on Apple Silicon
            ).eval()
            self.processor = ColQwen2Processor.from_pretrained(model)
        elif 'vidore' in model and 'pali' in model:
            self.embed_model = ColPali.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map=device,  # or "mps" if on Apple Silicon
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(model)
        elif 'bge-m3' in model:
            from FlagEmbedding import BGEM3FlagModel
            self.embed_model = BGEM3FlagModel(model, use_fp16=True, devices=[device])
        elif 'clip' in model:
            from transformers import CLIPProcessor, CLIPModel
            self.embed_model = CLIPModel.from_pretrained(model).to(device)
            self.processor = CLIPProcessor.from_pretrained(model)
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        elif 'e5-v' in model:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            self.processor = LlavaNextProcessor.from_pretrained(model)
            self.embed_model = LlavaNextForConditionalGeneration.from_pretrained(model, torch_dtype=torch.float16).to(device)
        elif 'gme' in model:
            self.embed_model = AutoModel.from_pretrained(model, torch_dtype="float16", device_map=device, trust_remote_code=True)
        elif 'dse' in model:
            from transformers import AutoProcessor, AutoModelForCausalLM
            self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
            self.embed_model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.float16).to(device)

    @classmethod
    def class_name(cls) -> str:
        return "VL_Embedding"
    
    def embed_img(self, img_path):
        if isinstance(img_path, str):
            img_path = [img_path]
        if 'vidore' in self.model:
            images = [fix(Image.open(img)) for img in img_path]
            batch_images = self.processor.process_images(images).to(self.embed_model.device)
            with torch.no_grad():
                image_embeddings = self.embed_model(**batch_images)
        elif 'clip' in self.model:
            images = [fix(Image.open(img)) for img in img_path]
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                image_features = self.embed_model.get_image_features(**inputs)
            image_embeddings = image_features / image_features.norm(dim=-1, keepdim=True)

        elif 'dse' in self.model:
            images = [fix(Image.open(img)) for img in img_path]
            passage_prompts = [f"<|image_{idx}|>\nWhat is shown in this image?</s>" for idx,_ in enumerate(img_path, 1)]
            # Process inputs and get embeddings
            passage_inputs = self.processor(passage_prompts, images=images, return_tensors="pt", padding="longest",
                                       max_length=4096, truncation=True).to(self.embed_model.device)
            passage_inputs['input_ids'] = passage_inputs['input_ids'].squeeze(0)
            passage_inputs['attention_mask'] = passage_inputs['attention_mask'].squeeze(0)
            # passage_inputs['image_sizes'] = passage_inputs['image_sizes'].squeeze(0)
            with torch.no_grad():
                output = self.embed_model(**passage_inputs, return_dict=True, output_hidden_states=True)
            image_embeddings = get_embedding(output.hidden_states[-1], passage_inputs["attention_mask"])

        elif 'gme' in self.model:
            images = [fix(Image.open(img)) for img in img_path]
            image_embeddings = self.embed_model.get_image_embeddings(images=images)

        elif 'e5-v' in self.model:
            images = [fix(Image.open(img)) for img in img_path]
            ps = self.embed_model.config.vision_config.patch_size  # 14
            self.processor.patch_size = ps

            # （可选但推荐）如果 image_processor 上也没有 patch_size，就顺便补上
            if getattr(self.processor.image_processor, "patch_size", None) is None:
                setattr(self.processor.image_processor, "patch_size", ps)
            llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
            img_prompt = llama3_template.format('<image>\nSummary above image in one word: ')
            img_inputs = self.processor(text=[img_prompt] * len(images), images=images, return_tensors="pt", padding=True).to(self.embed_model.device)
            with torch.no_grad():
                img_embs = self.embed_model(**img_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                image_embeddings = F.normalize(img_embs, dim=-1)

        elif 'openbmb' in self.model:
            images = [Image.open(img).convert('RGB') for img in img_path]
            inputs = {
                "text": [''] * len(images),
                'image': images,
                'tokenizer': self.tokenizer
            }
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                attention_mask = outputs.attention_mask
                hidden = outputs.last_hidden_state
                reps = weighted_mean_pooling(hidden, attention_mask)   
                image_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
                # image_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().tolist()[0]
            # image_embeddings = embeddings.tolist()[0]
        return image_embeddings
    
    def embed_text(self, text):
        if isinstance(text, str):
            text = [text]
        if 'colqwen' in self.model:
            batch_queries = self.processor.process_queries(text).to(self.embed_model.device)
            with torch.no_grad():
                query_embeddings = self.embed_model(**batch_queries)
        elif 'colpali' in self.model:
            batch_queries = self.processor.process_queries(text).to(self.embed_model.device)
            with torch.no_grad():
                query_embeddings = self.embed_model(**batch_queries)
        elif 'openbmb' in self.model:
            INSTRUCTION = "Represent this query for retrieving relevant documents: "
            queries = [INSTRUCTION + query for query in text]
            inputs = {
                "text": queries,
                'image': [None] * len(queries),
                'tokenizer': self.tokenizer
                }
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                attention_mask = outputs.attention_mask
                hidden = outputs.last_hidden_state
                reps = weighted_mean_pooling(hidden, attention_mask)   
                # query_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
                query_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().tolist()
                # query_embeddings = embeddings.tolist()[0]
        return query_embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.embed_text(query)[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.embed_text(text)[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings_list: List[List[float]] = []
        for text in texts:
            embeddings = self.embed_text(text)
            embeddings = embeddings[0]
            embeddings_list.append(embeddings)
        return embeddings_list

    def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.embed_text(query)[0]
    
    def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.embed_text(text)[0]
    
    def _get_image_embedding(self, img_file_path) -> Embedding:
        return self.embed_img(img_file_path)
    
    def _aget_image_embedding(self, img_file_path) -> Embedding:
        return self.embed_img(img_file_path)
    
    def __call__(self, nodes, **kwargs):
        if 'vidore' in self.model:
            if self.mode == 'image':
                embeddings = self.embed_img([node.metadata['file_path'] for node in nodes])
                embeddings = embeddings.view(embeddings.size(0),-1).tolist()
            else:
                embeddings = self.embed_text([node.text for node in nodes])
                embeddings = embeddings.view(embeddings.size(0),-1).tolist()

            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding

        elif 'clip' in self.model or 'e5-v' in self.model or 'gme' in self.model or 'dse' in self.model:
            embeddings = self.embed_img([node.metadata['file_path'] for node in nodes])
            embeddings = embeddings.view(embeddings.size(0), -1).tolist()
            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding

        elif 'bge-m3' in self.model:
            texts = [n.text for n in nodes]
            embeddings = self.embed_model.encode(texts)['dense_vecs']
            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding.tolist()

        elif 'openbmb' in self.model:
            if self.mode == 'image':
                embeddings = self.embed_img([node.metadata['file_path'] for node in nodes])
                embeddings = embeddings.tolist()
            else:
                embeddings = self.embed_text([node.text for node in nodes])
                # embeddings = embeddings.tolist()
                # embeddings = [embeddings]

            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding

        return nodes
    
    def score(self,image_embeddings,text_embeddings):
        if 'vidore' in self.model:
            score = self.processor.score_multi_vector(image_embeddings, text_embeddings)
        elif 'openbmb' in self.model:
            score = text_embeddings @ image_embeddings.T
        return score

if __name__ == "__main__":
    colpali = VL_Embedding("/data/user0/models/vidore/colqwen2-v1.0")
    image_embeddings = colpali.embed_img("../search_engine/corpus/img/welcome-to-nus_9.jpg")
    text_embeddings = colpali.embed_text("Hello, world!")
    score = colpali.processor.score_multi_vector(image_embeddings, text_embeddings)
    print(score)