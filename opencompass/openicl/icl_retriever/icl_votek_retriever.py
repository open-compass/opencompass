"""Votek Retriever."""

import json
import os
import random
from collections import defaultdict
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from opencompass.openicl.icl_retriever.icl_topk_retriever import TopkRetriever


class VotekRetriever(TopkRetriever):
    """Vote-k In-context Learning Retriever, subclass of `TopkRetriever`.

    **WARNING**: This class has not been tested thoroughly. Please use it with
    caution.
    """

    def __init__(self,
                 dataset,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 ice_num: Optional[int] = 1,
                 sentence_transformers_model_name: Optional[
                     str] = 'all-mpnet-base-v2',
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1,
                 votek_k: Optional[int] = 3) -> None:
        super().__init__(dataset, ice_separator, ice_eos_token, ice_num,
                         sentence_transformers_model_name, tokenizer_name,
                         batch_size)
        self.votek_k = votek_k

    def votek_select(self,
                     embeddings=None,
                     select_num=None,
                     k=None,
                     overlap_threshold=None,
                     vote_file=None):
        n = len(embeddings)
        if vote_file is not None and os.path.isfile(vote_file):
            with open(vote_file, encoding='utf-8') as f:
                vote_stat = json.load(f)
        else:
            vote_stat = defaultdict(list)

            for i in range(n):
                cur_emb = embeddings[i].reshape(1, -1)
                cur_scores = np.sum(cosine_similarity(embeddings, cur_emb),
                                    axis=1)
                sorted_indices = np.argsort(cur_scores).tolist()[-k - 1:-1]
                for idx in sorted_indices:
                    if idx != i:
                        vote_stat[idx].append(i)

            if vote_file is not None:
                with open(vote_file, 'w', encoding='utf-8') as f:
                    json.dump(vote_stat, f)
        votes = sorted(vote_stat.items(),
                       key=lambda x: len(x[1]),
                       reverse=True)
        j = 0
        selected_indices = []
        while len(selected_indices) < select_num and j < len(votes):
            candidate_set = set(votes[j][1])
            flag = True
            for pre in range(j):
                cur_set = set(votes[pre][1])
                if len(candidate_set.intersection(
                        cur_set)) >= overlap_threshold * len(candidate_set):
                    flag = False
                    break
            if not flag:
                j += 1
                continue
            selected_indices.append(int(votes[j][0]))
            j += 1
        if len(selected_indices) < select_num:
            unselected_indices = []
            cur_num = len(selected_indices)
            for i in range(n):
                if i not in selected_indices:
                    unselected_indices.append(i)
            selected_indices += random.sample(unselected_indices,
                                              select_num - cur_num)
        return selected_indices

    def vote_k_search(self):
        vote_k_idxs = self.votek_select(embeddings=self.embed_list,
                                        select_num=self.ice_num,
                                        k=self.votek_k,
                                        overlap_threshold=1)
        return [vote_k_idxs[:] for _ in range(len(self.test_ds))]

    def retrieve(self):
        return self.vote_k_search()
