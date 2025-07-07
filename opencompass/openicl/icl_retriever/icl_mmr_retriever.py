"""Topk Retriever."""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import tqdm
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from sklearn.metrics.pairwise import cosine_similarity

from opencompass.openicl.icl_dataset_reader import DatasetEncoder
from opencompass.openicl.icl_retriever import BaseRetriever
from opencompass.openicl.utils.logging import get_logger
from opencompass.registry import ICL_RETRIEVERS

logger = get_logger(__name__)


@ICL_RETRIEVERS.register_module()
class MMRRetriever(BaseRetriever):
    """Base class for Topk In-context Learning Retriever, implemented with
    basic knn. SentenceTransformer is used to calculate embeddings. Faiss is
    used to do the nearest neighbor search.

    Args:
        dataset (`BaseDataset`): Any BaseDataset instances.
            Attributes of ``reader``, ``train`` and ``test`` will be used.
        ice_separator (`Optional[str]`): The separator between each in-context
            example template when origin `PromptTemplate` is provided. Defaults
            to '\n'.
        ice_eos_token (`Optional[str]`): The end of sentence token for
            in-context example template when origin `PromptTemplate` is
            provided. Defaults to '\n'.
        ice_num (`Optional[int]`): The number of in-context example template
            when origin `PromptTemplate` is provided. Defaults to 1.
        sentence_transformers_model_name (`Optional[str]`): The name of the
            sentence transformers model. Defaults to 'all-mpnet-base-v2'.
        tokenizer_name (`Optional[str]`): The name of the tokenizer. Defaults
            to 'gpt2-xl'.
        batch_size (`Optional[int]`): The batch size for the dataloader.
            Defaults to 1.
    """
    model = None

    def __init__(self,
                 dataset,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 ice_num: Optional[int] = 8,
                 sentence_transformers_model_name: Optional[
                     str] = 'all-mpnet-base-v2',
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 10) -> None:
        super().__init__(dataset, ice_separator, ice_eos_token, ice_num)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        gen_datalist = self.dataset_reader.generate_input_field_corpus(
            self.test_ds)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'right'

        self.encode_dataset = DatasetEncoder(gen_datalist,
                                             tokenizer=self.tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer,
                                            device=self.device)
        self.dataloader = DataLoader(self.encode_dataset,
                                     batch_size=self.batch_size,
                                     collate_fn=co)

        self.model = SentenceTransformer(sentence_transformers_model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.index = self.create_index()

    def _standardize_normalize(self, cosine_similarities):
        """标准化并归一化相似度"""
        # normalize into 0-1 range
        cosine_sims_norm = (cosine_similarities - np.min(cosine_similarities)) / (
            np.max(cosine_similarities) - np.min(cosine_similarities))
        # standardize and shift by 0.5
        cosine_sims_norm = 0.5 + (cosine_sims_norm - np.mean(cosine_sims_norm)) / np.std(cosine_sims_norm)
        return cosine_sims_norm

    def _max_normalize(self, cosine_similarities):
        """最大值归一化"""
        cosine_sims_norm = np.copy(cosine_similarities)
        np.fill_diagonal(cosine_sims_norm, np.NaN)
        # normalize into 0-1 range
        cosine_sims_norm = (cosine_similarities - np.nanmin(cosine_similarities, axis=0)) / (
                np.nanmax(cosine_similarities, axis=0) - np.nanmin(cosine_similarities, axis=0))
        # standardize shift by 0.5
        cosine_sims_norm = \
            0.5 + (cosine_sims_norm - np.nanmean(cosine_sims_norm, axis=0)) / np.nanstd(cosine_sims_norm, axis=0)
        return cosine_sims_norm
    
    def create_index(self):
        import faiss

        self.select_datalist = self.dataset_reader.generate_input_field_corpus(
            self.index_ds)
        encode_datalist = DatasetEncoder(self.select_datalist,
                                         tokenizer=self.tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer,
                                            device=self.device)
        dataloader = DataLoader(encode_datalist,
                                batch_size=self.batch_size,
                                collate_fn=co)
        index = faiss.IndexIDMap(
            faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension()))
        res_list = self.forward(dataloader,
                                process_bar=True,
                                information='Creating index for index set...')
        id_list = np.array([res['metadata']['id'] for res in res_list])
        self.embed_list = np.stack([res['embed'] for res in res_list])
        index.add_with_ids(self.embed_list, id_list)
        return index
    
    def mmr_selection(self, query_embed, candidate_embeds, candidate_ids, k, beta=0.7):
        """MMR核心算法实现"""
        
        # 计算相似度矩阵
        text_sims = cosine_similarity(candidate_embeds, [query_embed]).flatten()
        candidate_sims = cosine_similarity(candidate_embeds)
        
        # 标准化处理
        text_sims_norm = self._standardize_normalize(text_sims)
        phrase_sims_norm = self._max_normalize(candidate_sims)
        
        selected_indices = []
        unselected_indices = list(range(len(candidate_embeds)))

        # 初始选择最相似样本
        best_idx = np.argmax(text_sims)
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)

        # 迭代选择剩余样本
        for _ in range(min(k-1, len(candidate_embeds)-1)):
            current_unselected = unselected_indices
            
            # 计算相关性和多样性得分
            relevance = text_sims_norm[current_unselected]
            diversity = np.max(phrase_sims_norm[np.ix_(current_unselected, selected_indices)], axis=1)
            
            # 计算MMR得分并选择最佳候选
            mmr_scores = beta * relevance - (1 - beta) * diversity
            best_rel_idx = np.argmax(mmr_scores)
            
            best_idx = current_unselected[best_rel_idx]
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)

        return [int(candidate_ids[idx]) for idx in selected_indices[:k]]
    
    def mmr_search(self, ice_num):
        res_list = self.forward(self.dataloader,
                                process_bar=True,
                                information='Embedding test set...')
        rtr_idx_list = [[] for _ in range(len(res_list))]
        
        # 获取全部候选embedding和对应的id
        candidate_embeds = self.embed_list
        candidate_ids = np.array([i for i in range(len(self.index_ds))])
        
        logger.info('Retrieving data with MMR...')
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry['metadata']['id']
            query_embed = entry['embed']
            
            # 使用MMR选择
            near_ids = self.mmr_selection(
                query_embed=query_embed,
                candidate_embeds=candidate_embeds,
                candidate_ids=candidate_ids,
                k=ice_num
            )
            rtr_idx_list[idx] = near_ids
        return rtr_idx_list

    def forward(self, dataloader, process_bar=False, information=''):
        res_list = []
        _dataloader = copy.deepcopy(dataloader)
        if process_bar:
            logger.info(information)
            _dataloader = tqdm.tqdm(_dataloader,
                                    disable=not self.is_main_process)
        for _, entry in enumerate(_dataloader):
            with torch.no_grad():
                metadata = entry.pop('metadata')
                raw_text = self.tokenizer.batch_decode(
                    entry['input_ids'],
                    skip_special_tokens=True,
                    verbose=False)
                res = self.model.encode(raw_text, show_progress_bar=False)
            res_list.extend([{
                'embed': r,
                'metadata': m
            } for r, m in zip(res, metadata)])
        return res_list

    def retrieve(self):
        """Retrieve the in-context example index for each test example."""
        return self.mmr_search(self.ice_num)


class ListWrapper:
    def __init__(self, data: List[Any]):
        self.data = data

    def to(self, device):
        return self.data

    def __iter__(self):
        return iter(self.data)


def ignore_pad_dict(features):
    res_dict = {}
    if 'metadata' in features[0]:
        res_dict['metadata'] = ListWrapper(
            [x.pop('metadata') for x in features])
    return res_dict


@dataclass
class DataCollatorWithPaddingAndCuda:
    tokenizer: PreTrainedTokenizerBase
    device: object = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 3000
    pad_to_multiple_of: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> BatchEncoding:
        res_dict = ignore_pad_dict(features)

        has_labels = 'labels' in features[0]
        if has_labels:
            labels = [{'input_ids': x.pop('labels')} for x in features]
            labels = self.tokenizer.pad(
                labels,
                padding=True,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors='pt',
                verbose=False)

        # print(features)
        batch = self.tokenizer.pad(features,
                                   padding=True,
                                   max_length=self.max_length,
                                   pad_to_multiple_of=self.pad_to_multiple_of,
                                   return_attention_mask=True,
                                   return_tensors='pt',
                                   verbose=False)

        if has_labels:
            batch['labels'] = labels.input_ids
        batch.update(res_dict)

        if self.device:
            batch = batch.to(self.device)

        return batch
