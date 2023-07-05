"""Random Retriever."""

from typing import Optional

import numpy as np
from tqdm import trange

from opencompass.openicl.icl_retriever import BaseRetriever
from opencompass.openicl.utils.logging import get_logger

logger = get_logger(__name__)


class RandomRetriever(BaseRetriever):
    """Random Retriever. Each in-context example of the test prompts is
    retrieved in a random way.

    **WARNING**: This class has not been tested thoroughly. Please use it with
    caution.
    """

    def __init__(self,
                 dataset,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 ice_num: Optional[int] = 1,
                 seed: Optional[int] = 43) -> None:
        super().__init__(dataset, ice_separator, ice_eos_token, ice_num)
        self.seed = seed

    def retrieve(self):
        np.random.seed(self.seed)
        num_idx = len(self.index_ds)
        rtr_idx_list = []
        logger.info('Retrieving data for test set...')
        for _ in trange(len(self.test_ds), disable=not self.is_main_process):
            idx_list = np.random.choice(num_idx, self.ice_num,
                                        replace=False).tolist()
            rtr_idx_list.append(idx_list)
        return rtr_idx_list
