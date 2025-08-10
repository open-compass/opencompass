"""Random Retriever."""

from typing import List, Optional

from tqdm import trange

from opencompass.openicl.icl_retriever import BaseRetriever
from opencompass.openicl.utils.logging import get_logger
from opencompass.registry import ICL_RETRIEVERS

logger = get_logger(__name__)


@ICL_RETRIEVERS.register_module()
class FixKRetriever(BaseRetriever):
    """Fix-K Retriever. Each in-context example of the test prompts is
    retrieved as the same K examples from the index set.

    Args:
        dataset (`BaseDataset`): Any BaseDataset instances.
            Attributes of ``reader``, ``train`` and ``test`` will be used.
        fix_id_list (List[int]): List of in-context example indices for every
            test prompts.
        ice_separator (`Optional[str]`): The separator between each in-context
            example template when origin `PromptTemplate` is provided. Defaults
            to '\n'.
        ice_eos_token (`Optional[str]`): The end of sentence token for
            in-context example template when origin `PromptTemplate` is
            provided. Defaults to '\n'.
        ice_num (`Optional[int]`): The number of in-context example template
            when origin `PromptTemplate` is provided. Defaults to 1.
    """

    def __init__(self,
                 dataset,
                 fix_id_list: List[int],
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 ice_num: Optional[int] = 1) -> None:
        super().__init__(dataset, ice_separator, ice_eos_token, ice_num)
        self.fix_id_list = fix_id_list

    def retrieve(self):
        """Retrieve the in-context example index for each test example."""
        num_idx = len(self.index_ds)
        for idx in self.fix_id_list:
            assert idx < num_idx, f'Index {idx} is out of range of {num_idx}'
        rtr_idx_list = []
        for _ in trange(len(self.test_ds), disable=not self.is_main_process):
            rtr_idx_list.append(self.fix_id_list)
        return rtr_idx_list
