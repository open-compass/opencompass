# flake8: noqa
import json
import os.path as osp
import re
from collections import defaultdict

from datasets import Dataset

from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .utils import get_judgeanswer_and_reference

base_prompt = """Evaluate the Response based on the Query and criteria provided.

** Criteria **
```{criteria}```

** Query **
```{question}```

** Response **
```{prediction}```

Provide your evaluation based on the criteria:

```{criteria}```

Provide reasons for each score, indicating where and why any strengths or deficiencies occur within the Response. Reference specific passages or elements from the text to support your justification.
Ensure that each reason is concrete, with explicit references to the text that aligns with the criteria requirements.

Scoring Range: Assign an integer score between 1 to 10

** Output format **
Return the results in the following JSON format, Only output this JSON format and nothing else:
```json
{{
    "score": an integer score between 1 to 10,
    "reason": "Specific and detailed justification for the score using text elements."
}}
```
"""


@LOAD_DATASET.register_module()
class WritingBenchDataset(BaseDataset):

    def load(self, path: str, name: str, *args, **kwargs):
        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{name}.jsonl')
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                domain1 = data['domain1']
                domain2 = data['domain2']
                query = data['query']
                criteria = data['criteria']
                judge_prompt_list = []
                for criteria_item in criteria:
                    temp_prompt = base_prompt.format(question=query,
                                                     criteria=criteria_item,
                                                     prediction='{prediction}')
                    judge_prompt_list.append(temp_prompt)
                idx = data['index']
                raw_data.append({
                    'question': query,
                    'judge': {
                        'index': idx,
                        'domain1': domain1,
                        'domain2': domain2,
                        'query': query,
                        'judge_prompt_list': judge_prompt_list
                    }
                })
        dataset = Dataset.from_list(raw_data)
        return dataset


def post_process_writingbench(judgement: dict):
    """Input a string like below:

    {"score": 9, "reason": "The response provides..."}, and extract the score
    """
    match = re.search(r"[\"']score[\"']:\s*([0-9]+)", judgement['prediction'])
    if match:
        score = int(match.group(1))
    else:
        return None

    return {'score': score}


@DICT_POSTPROCESSORS.register_module('writingbench')
def writingbench_postprocess(output: dict, output_path: str) -> dict:
    judged_answers, references = get_judgeanswer_and_reference(
        output, output_path, post_process_writingbench)

    if len(judged_answers) == 0:
        scores = None

    scores = defaultdict(list)
    for ans, ref in zip(judged_answers, references):
        domain = ref['domain1']
        score = ans['score']
        if score is not None:
            scores['overall'].append(score)
            scores[domain].append(score)
    single_model_scores = {
        task: sum(score) / len(score)
        for task, score in scores.items()
    }
    results = single_model_scores
    results['details'] = output
    return results
