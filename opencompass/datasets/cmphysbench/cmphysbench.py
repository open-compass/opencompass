import re

from datasets import Dataset, load_dataset

from opencompass.datasets.cmphysbench.SEED.SEED import SEED
from opencompass.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class CMPhysBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = load_dataset(path, split='train')
        new_data = []
        for ins in dataset:
            if len(ins['final_answer']) == 0:
                continue
            new_data.append({
                'prompt':
                ins['question'],
                'ground_truth': [ins['final_answer'][0], ins['topic']],
            })
        dataset = Dataset.from_list(new_data)
        return dataset


def extract_boxed_text_overlap(text):
    """
    简化版本：如果只需要基本的boxed内容提取
    不做过多的LaTeX清理
    """
    boxed_positions = []
    for match in re.finditer(r'\\boxed\{', text):
        boxed_positions.append(match.start())

    if not boxed_positions:
        return None

    boxed_contents = []

    for start_pos in boxed_positions:
        brace_start = text.find('{', start_pos) + 1
        if brace_start == 0:
            continue

        brace_count = 1
        current_pos = brace_start

        while current_pos < len(text) and brace_count > 0:
            if text[current_pos] == '{':
                brace_count += 1
            elif text[current_pos] == '}':
                brace_count -= 1
            current_pos += 1

        if brace_count == 0:
            content = text[brace_start:current_pos - 1]
            boxed_contents.append(content)

    return boxed_contents[-1].strip() if boxed_contents else None


def extract_boxed_text_improved(text):
    """
    提取LaTeX文本中\boxed{}内容的改进版本
    能够处理复杂的嵌套大括号结构
    """
    # 找到所有\boxed{的位置
    boxed_positions = []
    for match in re.finditer(r'\\boxed\{', text):
        boxed_positions.append(match.start())

    if not boxed_positions:
        return None

    # 对每个\boxed{位置，找到对应的结束}
    boxed_contents = []

    for start_pos in boxed_positions:
        # 从\boxed{后开始计数大括号
        brace_start = text.find('{', start_pos) + 1
        if brace_start == 0:  # 没找到{
            continue

        brace_count = 1
        current_pos = brace_start

        # 逐字符扫描，正确匹配大括号
        while current_pos < len(text) and brace_count > 0:
            char = text[current_pos]

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

            current_pos += 1

        if brace_count == 0:  # 找到了匹配的}
            # 提取内容（不包括最后的}）
            content = text[brace_start:current_pos - 1]
            boxed_contents.append(content)

    if not boxed_contents:
        return None

    # 取最后一个匹配的内容
    boxed_content = boxed_contents[-1].strip()

    # 清理LaTeX命令
    # 1. 移除\text{}包装（只在完整匹配时）
    clean_content = re.sub(r'\\text\{([^}]*)\}', r'\1', boxed_content)

    # 2. 处理常见的LaTeX转义符（更保守的处理）
    # 只处理简单的转义，避免破坏复杂的LaTeX命令
    escape_patterns = [
        (r'\\&', '&'),
        (r'\\%', '%'),
        (r'\\\$', '$'),
        (r'\\#', '#'),
        (r'\\_', '_'),
        (r'\\textbackslash', '\\'),
    ]

    for pattern, replacement in escape_patterns:
        clean_content = re.sub(pattern, replacement, clean_content)

    return clean_content


@ICL_EVALUATORS.register_module()
class CMPhysBenchEvaluator(BaseEvaluator):
    """F1 score for fasta."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        details = []
        for prediction, ans in zip(predictions, references):
            pred = extract_boxed_text_overlap(prediction)
            if not pred:
                pred = prediction
            detail = {'pred': pred, 'answer': ans}
            score, _, _, _ = SEED(pred, ans[0], ans[1])
            detail['score'] = score
            details.append(detail)

        score = sum(detail['score']
                    for detail in details) / len(details) if details else 0.0
        return {'score': score, 'details': details}
