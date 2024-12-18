# LiveMathBench

## Details of Datsets

| dataset | language | #single-choice | #multiple-choice | #fill-in-the-blank | #problem-solving |
| -- | -- | -- | -- | -- | -- |
| AIMC | cn | 0 | 0 | 0 | 46 |
| AIMC | en | 0 | 0 | 0 | 46 |
| CEE | cn | 0 | 0 | 13 | 40 |
| CEE | en | 0 | 0 | 13 | 40 |
| CMO | cn | 0 | 0 | 0 | 18 |
| CMO | en | 0 | 0 | 0 | 18 |
| MATH500 | en | 0 | 0 | 0 | 500 |
| AIME2024 | en | 0 | 0 | 0 | 44 |


## How to use


```python
from mmengine.config import read_base

with read_base():
    from opencompass.datasets.livemathbench import livemathbench_datasets

livemathbench_datasets[0].update(
    {
        'abbr': 'livemathbench_${k}x${n}'
        'path': '/path/to/data/dir', 
        'k': 'k@pass', # the max value of k in k@pass
        'n': 'number of runs', # number of runs
    }
)
livemathbench_datasets[0]['eval_cfg']['evaluator'].update(
    {
        'model_name': 'Qwen/Qwen2.5-72B-Instruct', 
        'url': [
            'http://0.0.0.0:23333/v1', 
            '...'
        ]  # set url of evaluation models
    }
)

```

> ❗️ At present, `extract_from_boxed` is used to extract answers from model responses, and one can also leverage LLM for extracting through the following parameters, but this part of the code has not been tested.

```python
livemathbench_datasets[0]['eval_cfg']['evaluator'].update(
    {
        'model_name': 'Qwen/Qwen2.5-72B-Instruct', 
        'url': [
            'http://0.0.0.0:23333/v1', 
            '...'
        ],  # set url of evaluation models

        # for LLM-based extraction
        'use_extract_model': True,
        'post_model_name': 'oc-extractor',
        'post_url': [
            'http://0.0.0.0:21006/v1,
            '...'
        ]
    }
)
```

## Output Samples

| dataset | version | metric | mode | Qwen2.5-72B-Instruct |
|----- | ----- | ----- | ----- | -----|
| LiveMathBench | caed8f | 1@pass | gen | 26.07 |
| LiveMathBench | caed8f | 1@pass/std | gen | xx.xx |
| LiveMathBench | caed8f | 2@pass | gen | xx.xx |
| LiveMathBench | caed8f | 2@pass/std | gen | xx.xx |
| LiveMathBench | caed8f | pass-rate | gen | xx.xx |

