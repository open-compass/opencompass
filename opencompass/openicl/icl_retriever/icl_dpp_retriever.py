"""DPP Retriever."""

import math
from typing import Optional

import numpy as np
import tqdm

from opencompass.openicl.icl_retriever.icl_topk_retriever import TopkRetriever
from opencompass.openicl.utils.logging import get_logger

logger = get_logger(__name__)


class DPPRetriever(TopkRetriever):
    """DPP In-context Learning Retriever, subclass of `TopkRetriever`. Two-
    stage DPP is used, where first stage is to get results of TopK to reduce
    candidate sets. Chechout https://arxiv.org/abs/2302.05698 for details.

    **WARNING**: This class has not been tested thoroughly. Please use it with
    caution.
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
                 batch_size: Optional[int] = 1,
                 candidate_num: Optional[int] = 1,
                 seed: Optional[int] = 1,
                 scale_factor: Optional[float] = 0.1) -> None:
        super().__init__(dataset, ice_separator, ice_eos_token, ice_num,
                         sentence_transformers_model_name, tokenizer_name,
                         batch_size)
        self.candidate_num = candidate_num
        self.seed = seed
        self.scale_factor = scale_factor

    def dpp_search(self):
        res_list = self.forward(self.dataloader,
                                process_bar=True,
                                information='Embedding test set...')
        rtr_idx_list = [[] for _ in range(len(res_list))]
        logger.info('Retrieving data for test set...')
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry['metadata']['id']

            # get TopK results
            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = np.array(
                self.index.search(embed, self.candidate_num)[1][0].tolist())

            # DPP stage
            near_reps, rel_scores, kernel_matrix = self.get_kernel(
                embed, near_ids.tolist())

            # MAP inference
            samples_ids = fast_map_dpp(kernel_matrix, self.ice_num)

            # ordered by relevance score
            samples_scores = np.array([rel_scores[i] for i in samples_ids])
            samples_ids = samples_ids[(-samples_scores).argsort()].tolist()
            rtr_sub_list = [int(near_ids[i]) for i in samples_ids]

            rtr_idx_list[idx] = rtr_sub_list

        return rtr_idx_list

    def retrieve(self):
        return self.dpp_search()

    def get_kernel(self, embed, candidates):
        near_reps = np.stack(
            [self.index.index.reconstruct(i) for i in candidates], axis=0)
        # normalize first
        embed = embed / np.linalg.norm(embed)
        near_reps = near_reps / np.linalg.norm(
            near_reps, keepdims=True, axis=1)

        # to make kernel-matrix non-negative
        rel_scores = np.matmul(embed, near_reps.T)[0]
        rel_scores = (rel_scores + 1) / 2

        # to prevent overflow error
        rel_scores -= rel_scores.max()

        # to balance relevance and diversity
        rel_scores = np.exp(rel_scores / (2 * self.scale_factor))

        # to make kernel-matrix non-negative
        sim_matrix = np.matmul(near_reps, near_reps.T)
        sim_matrix = (sim_matrix + 1) / 2

        kernel_matrix = rel_scores[None] * sim_matrix * rel_scores[:, None]
        return near_reps, rel_scores, kernel_matrix


def fast_map_dpp(kernel_matrix, max_length):
    """fast implementation of the greedy algorithm reference:

    https://github.com/laming-chen/fast-map-dpp/blob/master/dpp_test.py
    paper: Fast Greedy MAP Inference for Determinantal Point Process to Improve
    Recommendation Diversity
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(int(selected_item))
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        selected_items.append(int(selected_item))
    return selected_items
