"""Zeroshot Retriever."""

from typing import List, Optional

from opencompass.openicl.icl_retriever import BaseRetriever
from opencompass.registry import ICL_RETRIEVERS
from opencompass.utils.logging import get_logger


@ICL_RETRIEVERS.register_module()
class ZeroRetriever(BaseRetriever):
    """Zeroshot Retriever. The retriever returns empty list for all queries.

    Args:
        dataset (`BaseDataset`): Any BaseDataset instances.
            Attributes of ``reader``, ``train`` and ``test`` will be used.
        ice_eos_token (`Optional[str]`): The end of sentence token for
            in-context example template when origin `PromptTemplate` is
            provided. Defaults to ''.
    """

    def __init__(self, dataset, ice_eos_token: Optional[str] = '') -> None:
        super().__init__(dataset, '', ice_eos_token, 0)

    def retrieve(self, id_list: List[int] = None) -> List[List]:
        if id_list is not None:
            get_logger().warning('id_list is not empty, but will be ignored.')
        rtr_idx_list = [[] for _ in range(len(self.test_ds))]
        return rtr_idx_list
