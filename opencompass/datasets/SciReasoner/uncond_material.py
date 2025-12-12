import re
from typing import Union

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS


@LOAD_DATASET.register_module()
class Uncond_material_Dataset(BaseDataset):

    @staticmethod
    def load(num, prompt):
        dataset = [{'input': prompt, 'output': ''} for _ in range(num)]
        return Dataset.from_list(dataset)


@TEXT_POSTPROCESSORS.register_module()
def material_postprocessor(text: Union[str, None]) -> str:
    if not text:
        return ''

    match = re.search(r'<material>(.*?)</material>', text,
                      re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return ''


class uncond_material_Evaluator(BaseEvaluator):

    def score(self, predictions):
        total = len(predictions)
        format_valid = 0
        smact_valid = 0
        from collections import Counter

        from smact.screening import smact_validity
        for text in predictions:

            match = re.match(
                r'([A-Z][a-z]?(?: [A-Z][a-z]?)*?)'
                r'\s*(?:<|⟨)sg(?:>|⟩)\s*(?:<|⟨)sg(\d+)(?:>|⟩)', text.strip())
            if not match:
                continue

            elements_str, sg_num = match.groups()
            elements = elements_str.split()
            counter = Counter(elements)
            formula = ''
            for el, cnt in sorted(counter.items()):
                formula += el
                if cnt > 1:
                    formula += str(cnt)
            try:
                if smact_validity(formula):
                    smact_valid += 1
                format_valid += 1
            except Exception:
                continue

        smact_validity_ratio_in_format_valid = smact_valid / format_valid \
            if format_valid else 0
        smact_validity_ratio_in_all = smact_valid / total if total else 0

        return {
            'total_samples':
            total,
            'format_valid_count':
            format_valid,
            'smact_valid_count':
            smact_valid,
            'smact_validity_ratio_in_format_valid':
            smact_validity_ratio_in_format_valid * 100,
            'smact_validity_ratio_in_all':
            smact_validity_ratio_in_all * 100,
        }
