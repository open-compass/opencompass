import json
import re

from opencompass.registry import TEXT_POSTPROCESSORS


def iter_jsonl(path):
    with open(path, 'r') as f:
        for line in f:
            yield json.loads(line)


def get_single_answer(item: dict):
    """Return the label of a row whose ``answer`` holds a single value.

    The official InfiniteBench data stores these answers as a one-element
    list, e.g. ``"answer": ["71432"]``, and the official ``compute_scores.py``
    scores against ``label[0]``. A plain value is returned unchanged.
    """
    answer = item['answer']
    if isinstance(answer, list):
        answer = answer[0]
    return answer


@TEXT_POSTPROCESSORS.register_module()
def InfiniteBench_first_number_postprocess(text: str) -> str:
    first_number = re.search(r'\d+\.\d+|\d+', text)
    if first_number is None:
        return None
    first_number = first_number.group(0).strip()
    return str(first_number)
