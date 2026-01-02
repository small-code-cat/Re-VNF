import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Mapping, Any, Dict
import json

import numpy as np
from tqdm import tqdm
import torch

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, ImageNode
from search_engine_utils.vl_embedding import VL_Embedding, get_embedding


def nodefile2node(input_file):
    nodes = []
    for doc in json.load(open(input_file, 'r')):
        if doc['class_name'] == 'TextNode' and doc['text'] != '':
            nodes.append(TextNode.from_dict(doc))
        elif doc['class_name'] == 'ImageNode':
            nodes.append(ImageNode.from_dict(doc))
        else:
            continue
    return nodes

class SearchEngine:
    def __init__(self, dataset_dir='search_engine/corpus', node_dir_prefix='colqwen_ingestion',embed_model_name='vidore/colqwen2-v1.0'): # "vidore/colqwen2-v0.1"

        self.workers = 1

        self.dataset_dir = dataset_dir

        self.node_dir = os.path.join(self.dataset_dir, node_dir_prefix)
        self.vector_embed_model = VL_Embedding(model=embed_model_name, mode='image')
        self.query_engine = self.load_query_engine()

    def load_nodes(self):
        files = os.listdir(self.node_dir)
        parsed_files = []
        max_workers = 128
        if max_workers == 1:
            for file in tqdm(files):
                input_file = os.path.join(self.node_dir, file)
                suffix = input_file.split('.')[-1]
                if suffix != 'node':
                    continue
                nodes = nodefile2node(input_file)
                parsed_files.extend(nodes)
        else:
            def parse_file(file,node_dir):
                input_file = os.path.join(node_dir, file)
                suffix = input_file.split('.')[-1]
                if suffix != 'node':
                    return []
                return nodefile2node(input_file)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # results = list(tqdm(executor.map(parse_file, files, self.node_dir), total=len(files)))
                results = list(tqdm(executor.map(parse_file, files, [self.node_dir]*len(files)), total=len(files)))
            # 合并所有线程的结果
            for result in results:
                parsed_files.extend(result)
        return parsed_files
        
    def load_query_engine(self):
        print('Loading nodes...')
        self.nodes = self.load_nodes()
        if 'bge' in self.node_dir:
            self.embedding_img = np.array([node.embedding for node in tqdm(self.nodes, desc="Creating & Moving Embeddings")])
        else:
            dim = 128
            if 'clip' in self.node_dir:
                dim = 512
            elif 'e5-v' in self.node_dir:
                dim = 4096
            elif 'gme' in self.node_dir:
                dim = 1536
            elif 'dse' in self.node_dir:
                dim = 3072
            self.embedding_img = [
                torch.tensor(node.embedding)
                .view(-1, dim)
                .bfloat16()
                .to(self.vector_embed_model.device)
                for node in tqdm(self.nodes, desc="Creating & Moving Embeddings")
            ]
        self.image_nums = len(self.embedding_img)

    def load_node_postprocessors(self):
        return []
    def batch_search(self, queries: List[str], top_k=20):
        if 'bge' in self.node_dir:
            query_embeddings = self.vector_embed_model.embed_model.encode(queries)['dense_vecs']
            similarity = query_embeddings @ self.embedding_img.T
            indices = np.argsort(-similarity, axis=1)[:, :top_k]
            topk_val = np.take_along_axis(similarity, indices, axis=1)
            recall_results = [[self.nodes[idx].text for idx in row] for row in indices]
        elif 'clip' in self.node_dir:
            inputs = self.vector_embed_model.tokenizer(queries, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.vector_embed_model.embed_model.device)
            with torch.inference_mode():
                text_features = self.vector_embed_model.embed_model.get_text_features(**inputs)
            text_embeddings = text_features / text_features.norm(dim=-1, keepdim=True).float()
            image_embs = torch.cat(self.embedding_img, dim=0).float()
            similarity = text_embeddings @ image_embs.T
            values, indices = torch.topk(similarity, k=min(self.image_nums, top_k), dim=1)
            recall_results = [[self.nodes[idx].metadata['file_name'] for idx in row] for row in indices]
        elif 'dse' in self.node_dir:
            image_embs = torch.cat(self.embedding_img, dim=0).float()
            query_inputs = self.vector_embed_model.processor(queries, return_tensors="pt", padding="longest", max_length=128,
                                     truncation=True).to(self.vector_embed_model.embed_model.device)
            with torch.no_grad():
                output = self.vector_embed_model.embed_model(**query_inputs, return_dict=True, output_hidden_states=True)
            query_embeddings = get_embedding(output.hidden_states[-1], query_inputs["attention_mask"]).float()
            similarity = query_embeddings @ image_embs.T
            values, indices = torch.topk(similarity, k=min(self.image_nums, top_k), dim=1)
            recall_results = [[self.nodes[idx].metadata['file_name'] for idx in row] for row in indices]
        elif 'gme' in self.node_dir:
            image_embs = torch.cat(self.embedding_img, dim=0).float()
            e_text = self.vector_embed_model.embed_model.get_text_embeddings(texts=queries).float().to(self.vector_embed_model.embed_model.device)
            similarity = e_text @ image_embs.T
            values, indices = torch.topk(similarity, k=min(self.image_nums, top_k), dim=1)
            recall_results = [[self.nodes[idx].metadata['file_name'] for idx in row] for row in indices]
        elif 'e5-v' in self.node_dir:
            image_embs = torch.cat(self.embedding_img, dim=0).float()
            llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
            text_prompt = llama3_template.format('<sent>\nSummary above sentence in one word: ')
            text_inputs = self.vector_embed_model.processor(text=[text_prompt.replace('<sent>', text) for text in queries], return_tensors="pt",
                                    padding=True).to(self.vector_embed_model.embed_model.device)
            with torch.no_grad():
                text_embs = self.vector_embed_model.embed_model(**text_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][
                    :, -1, :]
                text_embs = torch.nn.functional.normalize(text_embs, dim=-1).float()
                similarity = text_embs @ image_embs.t()
            values, indices = torch.topk(similarity, k=min(self.image_nums, top_k), dim=1)
            recall_results = [[self.nodes[idx].metadata['file_name'] for idx in row] for row in indices]
        else:
            batch_queries = self.vector_embed_model.processor.process_queries(queries).to(self.vector_embed_model.embed_model.device)
            with torch.no_grad():
                query_embeddings = self.vector_embed_model.embed_model(**batch_queries)
            scores = self.vector_embed_model.processor.score_multi_vector(query_embeddings, self.embedding_img, batch_size=256, device=self.vector_embed_model.embed_model.device)
            values, indices = torch.topk(scores, k=min(self.image_nums,top_k), dim=1)
            recall_results = [[self.nodes[idx].metadata['file_name'] for idx in row] for row in indices]
        return recall_results

    def search_with_score(self, queries, top_k=20):
        batch_queries = self.vector_embed_model.processor.process_queries(queries).to(
            self.vector_embed_model.embed_model.device)
        with torch.no_grad():
            query_embeddings = self.vector_embed_model.embed_model(**batch_queries)
        scores = self.vector_embed_model.processor.score_multi_vector(query_embeddings, self.embedding_img,
                                                                      batch_size=256,
                                                                      device=self.vector_embed_model.embed_model.device)
        values, indices = torch.topk(scores, k=min(self.image_nums, top_k), dim=1)
        recall_results = [
            [
                (self.nodes[idx].metadata['file_name'], values[i, j].item())
                for j, idx in enumerate(row)
            ]
            for i, row in enumerate(indices)
        ]
        return recall_results

if __name__ == '__main__':
    search_engine = SearchEngine(dataset_dir='search_engine/corpus',embed_model_name='vidore/colqwen2-v1.0')
    print(search_engine.batch_search(['o','a']))
    

    