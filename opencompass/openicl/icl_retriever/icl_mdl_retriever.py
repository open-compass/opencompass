"""MDL Retriever."""

from typing import List, Optional

import numpy as np
import torch
import tqdm
from transformers import AutoModelForCausalLM

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever.icl_topk_retriever import TopkRetriever
from opencompass.openicl.utils.logging import get_logger
from opencompass.registry import ICL_PROMPT_TEMPLATES, ICL_RETRIEVERS

logger = get_logger(__name__)


@ICL_RETRIEVERS.register_module()
class MDLRetriever(TopkRetriever):
    """MDL Retriever, subclass of `TopkRetriever`. MDL is a abbreviation of
    Minimum Description Length, specially designed for ppl evaluation. You may
    refer to the paper for more details: https://arxiv.org/pdf/2212.10375.pdf.

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
        candidate_num (`Optional[int]`): The number of candidates to retrieve
            for each example. Defaults to 1.
        ce_model_name (`Optional[str]`): The name of the model for calculating
            MDL. Defaults to 'gpt2-xl'.
        select_time (`Optional[int]`): The number of times to select MDL.
            Defaults to 5.
        ice_template (`Optional[PromptTemplate]`): The template for in-context
            example. Defaults to None.
        prompt_template (`Optional[PromptTemplate]`): The template for prompt.
            Defaults to None.
        labels (`Optional[List]`): The labels for calculating MDL. Defaults to
            None.
        seed (`Optional[int]`): The seed for random. Defaults to 1.
    """
    metric_model = None

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
                 ce_model_name: Optional[str] = 'gpt2-xl',
                 select_time: Optional[int] = 5,
                 ice_template: Optional[PromptTemplate] = None,
                 prompt_template: Optional[PromptTemplate] = None,
                 labels: Optional[List] = None,
                 seed: Optional[int] = 1) -> None:
        super().__init__(dataset, ice_separator, ice_eos_token, ice_num,
                         sentence_transformers_model_name, tokenizer_name,
                         batch_size)
        self.ce_model_name = ce_model_name
        self.candidate_num = candidate_num
        self.select_time = select_time
        self.ice_template = ICL_PROMPT_TEMPLATES.build(ice_template)
        if prompt_template is not None:
            self.prompt_template = ICL_PROMPT_TEMPLATES.build(prompt_template)
        else:
            self.prompt_template = None
        self.labels = labels
        self.seed = seed

    def topk_search(self):
        np.random.seed(self.seed)
        res_list = self.forward(self.dataloader)
        rtr_idx_list = [[] for _ in range(len(res_list))]

        logger.info('Retrieving data for test set...')
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry['metadata']['id']
            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = self.index.search(
                embed, min(self.candidate_num,
                           len(self.index_ds)))[1][0].tolist()
            candidates = []
            mdl_scores = []
            for j in range(self.select_time):
                if j == 0:
                    rand_idx_list = near_ids[:self.ice_num]
                else:
                    rand_idx_list = np.random.choice(near_ids,
                                                     self.ice_num,
                                                     replace=False)
                    rand_idx_list = [int(i) for i in rand_idx_list]
                candidates.append(rand_idx_list)

                ice = self.generate_ice(rand_idx_list,
                                        ice_template=self.ice_template)
                ice = str(ice)
                mask_length = len(
                    self.tokenizer(ice + self.ice_eos_token,
                                   verbose=False)['input_ids'])
                if self.labels is None:
                    labels = self.get_labels(self.ice_template,
                                             self.prompt_template)
                else:
                    labels = self.labels
                prompt_list = []
                for label in labels:
                    prompt = self.generate_label_prompt(
                        idx, ice, label, self.ice_template,
                        self.prompt_template)
                    prompt = str(prompt)
                    prompt_list.append(prompt)
                loss_list = self.cal_ce(prompt_list, mask_length=mask_length)
                probs = np.exp(-np.array(loss_list))
                normalized_probs = probs / probs.sum(0, keepdims=True)
                neg_entropy = -entropy(normalized_probs, label_dim=0)
                mdl_scores.append(neg_entropy)

            rtr_idx_list[idx] = candidates[mdl_scores.index(max(mdl_scores))]
            rtr_idx_list[idx] = [int(i) for i in rtr_idx_list[idx]]

        return rtr_idx_list

    def retrieve(self):
        """Retrieve the in-context example index for each test example."""
        return self.topk_search()

    @torch.no_grad()
    def cal_ce(self, input_texts: List[str], mask_length=None):
        if self.metric_model is None:
            logger.info(
                f'Load model {self.ce_model_name} for calculating MDL...')
            self.metric_model = AutoModelForCausalLM.from_pretrained(
                self.ce_model_name)
            self.metric_model.to(self.device)
        inputs = self.tokenizer(input_texts,
                                padding=True,
                                return_tensors='pt',
                                truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.metric_model(**inputs)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs['input_ids'][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad_token_id)
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        loss = loss_fct(shift_logits,
                        shift_labels.view(-1)).view(shift_labels.size())
        if mask_length is not None:
            mask = torch.cat([
                torch.zeros([loss.shape[0], mask_length], dtype=torch.float),
                torch.ones([loss.shape[0], loss.shape[-1] - mask_length],
                           dtype=torch.float)
            ], -1)
            mask = mask.to(self.device)
            loss = torch.mul(mask, loss)

        lens = (inputs['input_ids'] !=
                self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= mask_length
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss


def entropy(probs: np.array, label_dim: int = 0, mask=None):
    if mask is None:
        return -(probs * np.log(probs)).sum(label_dim)
    return -(mask * probs * np.log(probs)).sum(label_dim)
