import json
import re
from collections import Counter
from typing import Union

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS


def extract_elements_from_prompt(prompt: str) -> list:
    """
    Extract element symbols from diverse prompt instructions.
    Supported patterns include:
    - composed of
    - that has
    - characterized by
    - with the composition
    - based on
    - featuring
    - whose makeup is
    """
    patterns = [
        r'composed of', r'that has', r'characterized by',
        r'with the composition', r'based on', r'featuring', r'whose makeup is'
    ]

    joined = '|'.join(patterns)
    match = re.search(rf'(?:{joined})\s+(.*?)(?:[\.。\n]|$)', prompt,
                      re.IGNORECASE)

    if match:
        elements_str = match.group(1)
        elements = [
            el.strip() for el in re.split(r'[,\s]+', elements_str)
            if re.fullmatch(r'[A-Z][a-z]?', el.strip())
        ]
        return elements

    # fallback: 尝试提取所有可能的元素符号
    fallback = re.findall(r'\b[A-Z][a-z]?\b', prompt)
    return fallback


def composition_precision(elements: list[str], prediction: str) -> float:
    """计算元素命中率"""
    E_pi = set(elements)
    clean = re.sub(r'<[^>]+>', ' ', prediction)
    E_gi = set(re.findall(r'\b[A-Z][a-z]?\b', clean))
    if not E_pi:
        return 0.0
    return len(E_pi & E_gi) / len(E_pi)


@LOAD_DATASET.register_module()
class Composition_material_Dataset(BaseDataset):

    @staticmethod
    def load(train_path, test_path, mini_set=False, hf_hub=False):
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

        # load from local json file
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        train_data = train_data[:5]
        # Limit the dataset to 5 samples for testing purposes
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


@TEXT_POSTPROCESSORS.register_module()
def material_postprocessor(text: Union[str, None]) -> str:
    """提取 <material> 标签内容"""
    if not text:
        return ''
    match = re.search(r'<material>(.*?)</material>', text,
                      re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ''


class composition_Evaluator(BaseEvaluator):

    def __init__(self, data_path, tuning_data=None, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.prompts = []
        self.gt_materials = set()

        if self.data_path:
            self._load_original_inputs()

    def _load_original_inputs(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.prompts = [item.get('input', '') for item in data]

        for item in data:
            output = item.get('output', '')
            mat = material_postprocessor(output)
            if mat:
                self.gt_materials.add(mat.strip())

    def _normalize(self, formula):
        tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
        tokens.sort(key=lambda x: x[0])
        return ''.join(f"{el}{cnt or ''}" for el, cnt in tokens)

    def score(self, predictions):
        from smact.screening import smact_validity

        total = len(predictions)
        format_valid = 0
        smact_valid = 0
        precision_sum = 0.0
        novel_count = 0

        for i, item in enumerate(predictions):
            if isinstance(item, str):
                item = {'prediction': item}

            text = item.get('prediction', '').strip()
            prompt = item.get('input', '').strip()
            if not prompt and i < len(self.prompts):
                prompt = self.prompts[i]

            prompt_elements = extract_elements_from_prompt(prompt)

            print('== Sample ==')
            print('Prompt:', prompt)
            print('Prompt Elements:', prompt_elements)
            print('Prediction Text:', text[:200])

            # --- SMAct validity ---
            match = re.match(
                r'([A-Z][a-z]?(?: [A-Z][a-z]?)*?)\s*(?:<sg>\s*)?<sg(\d+)>',
                text)
            if match:
                elements_str, _ = match.groups()
                elements = elements_str.split()
                counter = Counter(elements)
                formula = ''.join(f"{el}{cnt or ''}"
                                  for el, cnt in sorted(counter.items()))
                try:
                    if smact_validity(formula):
                        smact_valid += 1
                    format_valid += 1
                except Exception:
                    pass

            # --- Composition precision ---
            if prompt_elements:
                precision_sum += composition_precision(prompt_elements, text)

            # --- Novelty ---
            predicted_material = material_postprocessor(text)
            if not predicted_material:
                predicted_material = text.strip()

            if predicted_material:
                print(f'[Novelty Check] GT materials: {self.gt_materials}')
                print(
                    f'[Novelty Check] Predicted material: {predicted_material}'
                )
                if predicted_material not in self.gt_materials:
                    novel_count += 1
                    print('[Novelty] Novel')
                else:
                    print('[Novelty] Seen before')

        avg_precision = (precision_sum / total * 100) if total else 0.0
        smact_in_format = (smact_valid / format_valid *
                           100) if format_valid else 0.0
        smact_in_all = (smact_valid / total * 100) if total else 0.0
        novelty_ratio = (novel_count / total * 100) if total else 0.0

        return {
            'total_samples': total,
            'format_valid_count': format_valid,
            'smact_valid_count': smact_valid,
            'smact_validity_ratio_in_format_valid_%': smact_in_format,
            'smact_validity_ratio_in_all_%': smact_in_all,
            'average_precision_%': avg_precision,
            'novel_material_ratio_%': novelty_ratio,
        }
