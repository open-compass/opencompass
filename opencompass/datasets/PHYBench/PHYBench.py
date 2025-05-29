import re
from typing import Dict, List

import numpy as np
import sympy
from datasets import load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.datasets.PHYBench.EED.EED import EED
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET


@LOAD_DATASET.register_module()
class PHYBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = load_dataset(path, split='train')
        # only use first 100 examples
        return dataset.select(range(100))


def extract_last_latex(prediction: str) -> str:
    # 1) Find all \boxed{ occurrences and manually extract balanced content
    boxed_positions = [
        m.start() for m in re.finditer(r'\\boxed\s*\{', prediction)
    ]
    boxed_contents = []
    for pos in boxed_positions:
        # find the opening brace
        brace_start = prediction.find('{', pos)
        if brace_start == -1:
            continue
        # scan forward to find matching closing brace
        depth = 0
        for i in range(brace_start, len(prediction)):
            if prediction[i] == '{':
                depth += 1
            elif prediction[i] == '}':
                depth -= 1
                if depth == 0:
                    # extract between braces
                    boxed_contents.append(prediction[brace_start +
                                                     1:i].strip())
                    break

    if boxed_contents:
        return boxed_contents[-1]

    # 2) fallback: other delimiters
    cleaned = re.sub(r'^###.*$', '', prediction, flags=re.MULTILINE)
    cleaned = re.sub(r'[*\\-]{3,}', '', cleaned)
    cleaned = re.sub(r'(^|\n)[ \t]*[-*+] ', r'\1', cleaned)

    patterns = [
        r'\$\$(.*?)\$\$',
        r'\\\[(.*?)\\\]',
        r'\$(.*?)\$',
        r'\\\((.*?)\\\)',
    ]
    fragments = []
    for pat in patterns:
        for mm in re.finditer(pat, cleaned, re.DOTALL):
            fragments.append(mm.group(1).strip())
    if fragments:
        return fragments[-1]

    # 3) final fallback
    m2 = re.search(r'Final\s*Answer\s*:?\s*(.+)$', prediction, re.DOTALL)
    return m2.group(1).strip() if m2 else prediction.strip()


def _calculate_eed_score(pred_str: str, ref_str: str) -> float:
    """Calculate the Expression Edit Distance (EED) score.

    Args:
        pred_str (str): Predicted answer string (LaTeX format)
        ref_str (str): Reference answer string (LaTeX format)

    Returns:
        float: EED score between 0 and 100
    """
    try:
        # Normalize the inputs first
        # remove the first $$ and the last $$ from the ref_str

        clean_pred = extract_last_latex(pred_str)
        if '$$' in ref_str:
            clean_ref = ref_str.split('$$')[1].strip()
        else:
            clean_ref = extract_last_latex(ref_str)
        # only compare the rhs of rightmost =
        clean_pred = clean_pred.split('=')[-1].strip()
        clean_ref = clean_ref.split('=')[-1].strip()

        # try to convert the latex to sympy expression
        try:
            clean_pred_expr = sympy.latex(sympy.sympify(clean_pred))
            clean_ref_expr = sympy.latex(sympy.sympify(clean_ref))
        except Exception:
            clean_pred_expr = None
            clean_ref_expr = None
        eed_result = EED(clean_ref, clean_pred)
        if clean_pred_expr and clean_ref_expr:
            clean_eed_result = EED(clean_ref_expr, clean_pred_expr)
            final_eed_result = max(clean_eed_result[0], eed_result[0])
        else:
            final_eed_result = eed_result[0]
        return final_eed_result
    except Exception:
        return 0


@ICL_EVALUATORS.register_module()
class PHYBenchEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()

    def score(self,
              predictions: List[str],
              references: List[str],
              test_set: List[Dict] = None) -> Dict:
        """Evaluate predictions for PHYBench based on Accuracy and EED
        Score."""

        if len(predictions) != len(references):
            return {'error': 'Number of predictions and references mismatch.'}

        correct_count = 0
        total_count = len(predictions)
        eed_scores = []

        for idx, (pred_str, ref_str) in enumerate(zip(predictions,
                                                      references)):

            eed = _calculate_eed_score(pred_str, ref_str)
            eed_scores.append(eed)

            if abs(eed - 100) < 1e-6:
                correct_count += 1

        accuracy = (correct_count /
                    total_count) * 100 if total_count > 0 else 0
        average_eed_score = np.mean(eed_scores) if eed_scores else 0

        # Return results as a dictionary
        return {'accuracy': accuracy, 'eed_score': average_eed_score}
