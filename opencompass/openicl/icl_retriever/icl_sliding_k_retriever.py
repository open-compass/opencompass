"""Sliding Window Retriever."""

from typing import Optional

from tqdm import trange

from opencompass.openicl.icl_retriever import BaseRetriever
from opencompass.openicl.utils.logging import get_logger
from opencompass.registry import ICL_RETRIEVERS

logger = get_logger(__name__)


@ICL_RETRIEVERS.register_module()
class SlidingWindowRetriever(BaseRetriever):
    """Sliding Window Retriever. Each in-context example of the test prompts is
    retrieved based on a sliding window from the index set.

    Args:
        dataset (`BaseDataset`):
            Any BaseDataset instances.
            Attributes of ``reader``, ``train`` and ``test`` will be used.
        k (int):
            The number of in-context examples to retrieve for each test prompt.
        ice_separator (`Optional[str]`):
            The separator between each in-context
            example template when origin `PromptTemplate` is provided. Defaults
            to '\n'.
        ice_eos_token (`Optional[str]`):
            The end of sentence token for
            in-context example template when origin `PromptTemplate` is
            provided. Defaults to '\n'.
        ice_num (`Optional[int]`):
            The number of in-context example template
            when origin `PromptTemplate` is provided. Defaults to 1.
    """

    def __init__(self,
                 dataset,
                 k: int,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 ice_num: Optional[int] = 1) -> None:
        super().__init__(dataset, ice_separator, ice_eos_token, ice_num)
        self.k = k

    def retrieve(self):
        """Retrieve the in-context example index for each test example."""
        num_idx = len(self.index_ds)
        rtr_idx_list = []
        for current_index in trange(len(self.test_ds),
                                    disable=not self.is_main_process):
            if current_index < self.k:
                """For the first few examples, get the previous ones and pad
                with the last ones."""
                start_index = max(0, current_index - self.k)
                previous_shots = list(range(start_index, current_index))
                if len(previous_shots) < self.k:
                    pad_needed = self.k - len(previous_shots)
                    previous_shots = list(range(num_idx - pad_needed,
                                                num_idx)) + previous_shots
            else:
                # For other examples, retrieve the previous k examples
                previous_shots = list(
                    range(current_index - self.k, current_index))
            rtr_idx_list.append(previous_shots)
        return rtr_idx_list
