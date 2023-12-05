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

from opencompass.openicl.icl_dataset_reader import DatasetEncoder
from opencompass.openicl.icl_retriever import BaseRetriever
from opencompass.openicl.utils.logging import get_logger
from opencompass.registry import ICL_RETRIEVERS

logger = get_logger(__name__)


@ICL_RETRIEVERS.register_module()
class TopkRetriever(BaseRetriever):
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
                 ice_num: Optional[int] = 1,
                 sentence_transformers_model_name: Optional[
                     str] = 'all-mpnet-base-v2',
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1) -> None:
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

    def knn_search(self, ice_num):
        res_list = self.forward(self.dataloader,
                                process_bar=True,
                                information='Embedding test set...')
        rtr_idx_list = [[] for _ in range(len(res_list))]
        logger.info('Retrieving data for test set...')
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry['metadata']['id']
            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = self.index.search(embed, ice_num)[1][0].tolist()
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
        return self.knn_search(self.ice_num)


class ListWrapper:

    def __init__(self, data: List[Any]):
        self.data = data

    def to(self, device):
        return self.data


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
