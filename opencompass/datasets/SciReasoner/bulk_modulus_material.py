import json
import re
from collections import Counter
from typing import List, Union

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from smact.screening import smact_validity

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS


@LOAD_DATASET.register_module()
class Bulk_modulus_material_Dataset(BaseDataset):

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


class material_Evaluator(BaseEvaluator):
    """
    Evaluator for:
      - SMAct validity
      - Composition precision (based on output-extracted elements)
      - Exact match (between prediction and reference <material> block)
    """

    def __init__(self, data_path=None, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.prompt_elements_list = []  # 从 gt 提取的元素
        self.reference_materials = []  # exact match 的参考答案

        if self.data_path:
            self._load_ground_truths()

    def _load_ground_truths(self):
        """加载 ground truth 元素和材料"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            output = item.get('output', '')
            # 提取组成元素
            elements = re.findall(r'\b[A-Z][a-z]?\b',
                                  material_postprocessor(output))
            self.prompt_elements_list.append(elements)
            # 提取完整材料块用于 exact match
            self.reference_materials.append(material_postprocessor(output))

    def _normalize(self, formula: str) -> str:
        """标准化化学式（字母排序+数量）"""
        tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
        tokens.sort(key=lambda x: x[0])
        return ''.join(f"{el}{cnt or ''}" for el, cnt in tokens)

    def score(self, predictions: List[dict]):
        total = len(predictions)
        format_valid = 0
        smact_valid = 0
        precision_sum = 0.0
        exact_match_count = 0

        for i, item in enumerate(predictions):
            if isinstance(item, str):
                item = {'prediction': item}
            text = item.get('prediction', '').strip()

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
            if i < len(self.prompt_elements_list):
                gt_elements = set(self.prompt_elements_list[i])
                pred_elements = set(re.findall(r'\b[A-Z][a-z]?\b', text))
                correct = len(gt_elements & pred_elements)
                if gt_elements:
                    precision_sum += correct / len(gt_elements)

            # --- Exact Match ---
            if i < len(self.reference_materials):
                pred_mat = material_postprocessor(text)
                gt_mat = self.reference_materials[i]
                if pred_mat == gt_mat:
                    exact_match_count += 1

        avg_precision = (precision_sum / total * 100) if total else 0.0
        smact_in_format = (smact_valid / format_valid *
                           100) if format_valid else 0.0
        smact_in_all = (smact_valid / total * 100) if total else 0.0
        exact_match_ratio = (exact_match_count / total * 100) if total else 0.0

        return {
            'total_samples': total,
            'format_valid_count': format_valid,
            'smact_valid_count': smact_valid,
            'smact_validity_ratio_in_format_valid_%': smact_in_format,
            'smact_validity_ratio_in_all_%': smact_in_all,
            'average_precision_%': avg_precision,
            'exact_match_ratio_%': exact_match_ratio
        }
