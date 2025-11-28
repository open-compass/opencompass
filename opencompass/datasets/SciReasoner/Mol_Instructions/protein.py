# molecule task
# https://github.com/zjunlp/Mol-Instructions/tree/main/evaluation/molecule

import json
import re
from typing import List, Optional

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from mmengine.config import ConfigDict

from opencompass.datasets.base import BaseDataset
from opencompass.datasets.Mol_Instructions.normalized_SW_score import \
    normalized_smith_waterman
from opencompass.openicl import BaseEvaluator, RougeEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS


@LOAD_DATASET.register_module()
class Mol_Instructions_Dataset_Protein_Design(BaseDataset):

    @staticmethod
    def load(train_path, test_path, max_cut=-1, mini_set=False, hf_hub=False):
        # import pdb; pdb.set_trace()
        if (hf_hub is True):
            # load from huggingface hub
            train_data = []
            repo_id = test_path.split('/')[0] + '/' + test_path.split('/')[1]
            train_path = train_path.split(repo_id + '/')[1]
            test_path = test_path.split(repo_id + '/')[1]

            train_path = hf_hub_download(repo_id,
                                         train_path,
                                         repo_type='dataset')
            test_path = hf_hub_download(repo_id,
                                        test_path,
                                        repo_type='dataset')

        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        train_data = train_data[:5]
        # Limit the dataset to 5 samples for testing purposes

        if (max_cut != -1):
            test_data = test_data[:max_cut]
        if mini_set:
            import random
            random.seed(1024)
            test_data = random.sample(test_data, 150)
            random.seed()

        dataset = DatasetDict({
            'train': Dataset.from_list(train_data),
            'test': Dataset.from_list(test_data)
        })
        return dataset


@TEXT_POSTPROCESSORS.register_module('Mol_Instructions_postprocess_Protein')
def Mol_Instructions_postprocess_Protein(text, *args, **kwargs):
    """
    Filter end tokens in the sentences: "<|endoftext|>","<|im_end|>"
    """
    text = text.strip()
    text = re.sub(r'<\|endoftext\|>', '', text)
    text = re.sub(r'<\|im_end\|>', '', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'.*?</think>\s*', '', text, flags=re.DOTALL)
    text = text.strip()

    return text


class Mol_Instructions_Evaluator_Protein(RougeEvaluator):

    def __init__(self,
                 task='catalytic_activity',
                 pred_postprocessor: Optional[ConfigDict] = None):
        super().__init__(pred_postprocessor=pred_postprocessor, )
        self.task = task


@TEXT_POSTPROCESSORS.register_module(
    'Mol_Instructions_postprocess_Protein_Design')
def Mol_Instructions_postprocess_Protein_Design(text, *args, **kwargs):
    """
    Extract the protein str between  <protein> and </protein> in the sentences
    """
    text = text.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'.*?</think>\s*', '', text, flags=re.DOTALL)
    pattern = r'<protein>(.*?)</protein>'
    match = re.search(pattern, text)
    if match:
        text = match.group(1)
        valid_letters = set('ACDEFGHIKLMNPQRSTVWY')
        text = ''.join(filter(lambda x: x in valid_letters, text))
    else:
        text = ''
    return text


class Mol_Instructions_Evaluator_Protein_Design(BaseEvaluator):

    def __init__(self, task='protein_design', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task

    def score(self, predictions: List[str], references: List[str]):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        if not isinstance(predictions[0], list):
            predictions = [[pred] for pred in predictions]
        if not isinstance(references[0], list):
            references = [[ref] for ref in references]

        scores = []
        for pred, refer in zip(predictions, references):
            pred = pred[0].strip()
            refer = refer[0].strip()
            if not pred or not refer:
                scores.append(0.0)
            else:
                # Calculate the normalized Smith-Waterman score
                score = normalized_smith_waterman(
                    pred, refer) * 100  # Convert to percentage
                scores.append(score)

        averaged_valid_scores = [score for score in scores if score > 0]

        results = {
            'Max SW score':
            max(scores),
            'Min SW score':
            min(scores),
            'Average SW score':
            sum(scores) / len(scores),
            'valid average SW score':
            sum(averaged_valid_scores) /
            len(averaged_valid_scores) if averaged_valid_scores else 0.0,
        }
        return results
