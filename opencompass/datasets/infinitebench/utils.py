import json
import re

from opencompass.registry import TEXT_POSTPROCESSORS


def iter_jsonl(path):
    with open(path, 'r') as f:
        for line in f:
            yield json.loads(line)


@TEXT_POSTPROCESSORS.register_module()
def InfiniteBench_first_number_postprocess(text: str) -> str:
    first_number = re.search(r'\d+\.\d+|\d+', text)
    if first_number is None:
        return None
    first_number = first_number.group(0).strip()
    return str(first_number)
