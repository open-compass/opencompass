"""BM25 Retriever."""

from typing import List, Optional

import numpy as np
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from tqdm import trange

from opencompass.openicl.icl_retriever import BaseRetriever
from opencompass.openicl.utils.logging import get_logger
from opencompass.registry import ICL_RETRIEVERS

logger = get_logger(__name__)


@ICL_RETRIEVERS.register_module()
class BM25Retriever(BaseRetriever):
    """BM25 Retriever. In information retrieval, Okapi BM25 (BM is an
    abbreviation of best matching) is a ranking function used by search engines
    to estimate the relevance of documents to a given search query. You can
    find more details in https://en.wikipedia.org/wiki/Okapi_BM25. Each in-
    context example of the test prompts is retrieved by the BM25 Algorithm.

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
        index_split (`Optional[str]`): The split of the dataset to retrieve the
            in-context example index, used when `dataset_reader.dataset` is an
            instance of `datasets.Dataset`. Defaults to 'train'.
        test_split (`Optional[str]`): The split of the dataset to retrieve the
            in-context example, used when `dataset_reader.dataset` is an
            instance of `datasets.Dataset`. Defaults to 'test'.
    """
    bm25 = None
    index_corpus = None
    test_corpus = None

    def __init__(self,
                 dataset,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 ice_num: Optional[int] = 1) -> None:
        super().__init__(dataset, ice_separator, ice_eos_token, ice_num)
        self.index_corpus = [
            word_tokenize(data) for data in
            self.dataset_reader.generate_input_field_corpus(self.index_ds)
        ]
        self.bm25 = BM25Okapi(self.index_corpus)
        self.test_corpus = [
            word_tokenize(data) for data in
            self.dataset_reader.generate_input_field_corpus(self.test_ds)
        ]

    def retrieve(self) -> List[List]:
        """Retrieve the in-context example index for each test example."""
        rtr_idx_list = []
        logger.info('Retrieving data for test set...')
        for idx in trange(len(self.test_corpus),
                          disable=not self.is_main_process):
            query = self.test_corpus[idx]
            scores = self.bm25.get_scores(query)
            near_ids = list(np.argsort(scores)[::-1][:self.ice_num])
            near_ids = [int(a) for a in near_ids]
            rtr_idx_list.append(near_ids)
        return rtr_idx_list
