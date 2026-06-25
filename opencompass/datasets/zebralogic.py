import re

from datasets import load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class ZebraLogicDataset(BaseDataset):
    """ZebraLogic dataset for logical reasoning evaluation.

    Supports two modes:
    - mc_mode: Multiple-choice questions derived from zebra logic puzzles
      (3259 test samples). Each puzzle generates one question with 4-6 choices.
    - grid_mode: Full grid completion task (1000 test samples). Models must
      output the complete solution table.

    Paper: https://arxiv.org/abs/2502.01100
    HuggingFace: WildEval/ZebraLogic
    """

    @staticmethod
    def load(path: str, config: str = 'mc_mode'):
        dataset = load_dataset(path, config)
        ds = dataset['test']
        if config == 'mc_mode':
            # Format choices as a labeled list: (A) choice1 (B) choice2 ...
            def format_mc(example):
                labels = 'ABCDEFGHIJ'
                choices = example['choices']
                answer = example['answer']
                formatted_choices = ' '.join(f'({labels[i]}) {c}'
                                             for i, c in enumerate(choices))
                # Find the label for the correct answer
                if answer in choices:
                    answer_label = labels[choices.index(answer)]
                else:
                    answer_label = answer
                example['formatted_choices'] = formatted_choices
                example['answer_label'] = answer_label
                return example

            ds = ds.map(format_mc)
        dataset['train'] = ds
        dataset['test'] = ds
        return dataset


@ICL_EVALUATORS.register_module()
class ZebraLogicMCEvaluator(BaseEvaluator):
    """Evaluator for ZebraLogic mc_mode.

    Extracts the answer letter (A/B/C/...) from model output and compares
    with the ground truth answer label.
    """

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        details = []
        correct_count = 0

        for pred, ref in zip(predictions, references):
            extracted = _extract_mc_answer(pred)
            correct = extracted.upper() == str(ref).upper()
            details.append({
                'pred': pred,
                'extracted': extracted,
                'answer': ref,
                'correct': correct,
            })
            correct_count += int(correct)

        score = correct_count / len(predictions) * 100
        return {'score': score, 'details': details}


@ICL_EVALUATORS.register_module()
class ZebraLogicGridEvaluator(BaseEvaluator):
    """Evaluator for ZebraLogic grid_mode.

    Computes cell-level accuracy: what fraction of cells in the solution
    table the model gets correct.  The expected reference format is a dict
    or JSON-serialised dict with keys ``header`` (list of column names) and
    ``rows`` (list of lists), matching the dataset ``solution`` field.
    """

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        total_cells = 0
        correct_cells = 0
        fully_correct = 0
        details = []

        for pred, ref in zip(predictions, references):
            pred_text = str(pred)
            gold_rows = _parse_grid_reference(ref)
            pred_rows = _extract_grid_from_text(pred_text)

            if gold_rows is None:
                continue

            sample_total = sum(len(row) for row in gold_rows)
            sample_correct = 0

            if (pred_rows is not None and len(pred_rows) == len(gold_rows) + 1
                    and _is_header_row(pred_rows[0])):
                pred_rows = pred_rows[1:]

            if pred_rows is not None and len(pred_rows) == len(gold_rows):
                for gold_row, pred_row in zip(gold_rows, pred_rows):
                    for g, p in zip(gold_row, pred_row[:len(gold_row)]):
                        if _normalize_cell(g) == _normalize_cell(p):
                            sample_correct += 1
            else:
                # Try cell-by-cell matching with flattened text
                flat_gold = [c for row in gold_rows for c in row]
                for cell in flat_gold:
                    if re.search(re.escape(str(cell)), pred_text,
                                 re.IGNORECASE):
                        sample_correct += 1

            total_cells += sample_total
            correct_cells += sample_correct
            sample_acc = sample_correct / sample_total if sample_total else 0
            is_perfect = sample_correct == sample_total
            fully_correct += int(is_perfect)
            details.append({
                'pred': pred_text[:200],
                'answer': str(ref)[:200],
                'cell_accuracy': sample_acc,
                'correct': is_perfect,
            })

        cell_acc = correct_cells / total_cells * 100 if total_cells else 0
        perfect_acc = fully_correct / len(predictions) * 100
        return {
            'score': cell_acc,
            'cell_accuracy': cell_acc,
            'perfect_accuracy': perfect_acc,
            'details': details,
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _extract_mc_answer(text: str) -> str:
    """Extract the answer letter from model output.

    Tries several heuristics in order:
    1. "The answer is: (X)" or "The answer is: X"
    2. Last occurrence of a lone letter (A-J) on its own line
    3. The first letter found inside parentheses: (X)
    """
    text = str(text)

    # Pattern 1: explicit prefix
    m = re.search(r'[Tt]he (?:final )?answer is[:\s]+\(?([A-Ja-j])\)?', text)
    if m:
        return m.group(1).upper()

    # Pattern 2: **X** or boxed{X} style
    m = re.search(r'\*\*([A-Ja-j])\*\*', text)
    if m:
        return m.group(1).upper()

    # Pattern 3: last standalone letter on a line
    lines = text.strip().split('\n')
    for line in reversed(lines):
        m = re.match(r'^\s*\(?([A-Ja-j])\)?\s*$', line.strip())
        if m:
            return m.group(1).upper()

    # Pattern 4: (X) anywhere in text
    matches = re.findall(r'\(([A-Ja-j])\)', text)
    if matches:
        return matches[-1].upper()

    return ''


def _parse_grid_reference(ref: str):
    """Parse the solution field stored as a dict or string.

    The field has a 'rows' key containing a 2-D list. Returns a list of lists
    of strings, or None on failure.
    """
    if isinstance(ref, dict):
        return ref.get('rows', None)

    import ast
    try:
        data = ast.literal_eval(ref)
        return data.get('rows', None)
    except Exception:
        pass
    # Try JSON
    import json
    try:
        data = json.loads(ref)
        return data.get('rows', None)
    except Exception:
        return None


def _normalize_cell(cell) -> str:
    return str(cell).strip().lower()


def _is_header_row(row) -> bool:
    if not row:
        return False
    return _normalize_cell(row[0]) in {'house', 'houses', 'home', 'position'}


def _extract_grid_from_text(text: str):
    """Try to extract a markdown-style table from model output.

    Returns a list of lists of strings (excluding the header separator row),
    or None if no table is found.
    """
    rows = []
    for line in text.split('\n'):
        line = line.strip()
        if not line.startswith('|'):
            continue
        # Skip separator rows like |---|---|
        if re.match(r'^\|[-| :]+\|$', line):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if cells:
            rows.append(cells)
    return rows if len(rows) > 1 else None
