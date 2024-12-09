# LiveMathBench

## Details of Datsets

| dataset | language | #single-choice | #multiple-choice | #fill-in-the-blank | #problem-solving |
| -- | -- | -- | -- | -- | -- |
| AMC | cn | 0 | 0 | 0 | 46 |
| AMC | en | 0 | 0 | 0 | 46 |
| CCEE | cn | 0 | 0 | 13 | 40 |
| CCEE | en | 0 | 0 | 13 | 40 |
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
        'abbr': 'livemathbench_${k}x${n}_${EVAL_MODEL_ABBR}'
        'path': '/path/to/data/dir', 
        'k': 'k@pass', # the max value of k in k@pass
        'n': 'number of runs', # number of runs
    }
)
livemathbench_datasets[0]['eval_cfg']['evaluator'].update(
    {
        'model_name': 'Qwen/Qwen2.5-72B-Instruct', # By default, Qwen2.5-72B-Instruct is used to measure answer equality
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

| dataset | version | metric | mode | Skywork-o1-Open-Llama-3.1-8B |
|----- | ----- | ----- | ----- | -----|
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass-rate@1 | gen | 64.17 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass-rate@1/std | gen | 1.12 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass@1 | gen | 64.17 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass@1/std | gen | 1.12 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass-rate@1 | gen | 9.09 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass-rate@1/std | gen | 3.21 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass@1 | gen | 9.09 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass@1/std | gen | 3.21 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass-rate@1 | gen | 49.28 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass-rate@1/std | gen | 1.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass@1 | gen | 49.28 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass@1/std | gen | 1.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass-rate@1 | gen | 37.68 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass-rate@1/std | gen | 1.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass@1 | gen | 37.68 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass@1/std | gen | 1.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass-rate@1 | gen | 46.97 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass-rate@1/std | gen | 3.21 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass@1 | gen | 46.97 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass@1/std | gen | 3.21 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass-rate@1 | gen | 50.76 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass-rate@1/std | gen | 2.14 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass@1 | gen | 50.76 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass@1/std | gen | 2.14 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass-rate@1 | gen | 22.22 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass-rate@1/std | gen | 0.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass@1 | gen | 22.22 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass@1/std | gen | 0.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass-rate@1 | gen | 37.04 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass-rate@1/std | gen | 2.62 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass@1 | gen | 37.04 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass@1/std | gen | 2.62 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass-rate@1 | gen | 78.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass-rate@1/std | gen | 0.66 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass@1 | gen | 78.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass@1/std | gen | 0.66 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | 16-pass@0.5 | gen | 58.64 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | 16-pass@0.5/std | gen | 1.49 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | 16-pass@0.75 | gen | 55.79 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | 16-pass@0.75/std | gen | 1.05 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | 16-pass@1.0 | gen | 52.37 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | 16-pass@1.0/std | gen | 1.36 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/16-pass@0.5 | gen | 4.55 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/16-pass@0.5/std | gen | 0.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/16-pass@0.75 | gen | 4.55 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/16-pass@0.75/std | gen | 0.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/16-pass@1.0 | gen | 3.79 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/16-pass@1.0/std | gen | 1.07 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/16-pass@0.5 | gen | 40.58 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/16-pass@0.5/std | gen | 2.05 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/16-pass@0.75 | gen | 36.23 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/16-pass@0.75/std | gen | 2.05 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/16-pass@1.0 | gen | 30.43 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/16-pass@1.0/std | gen | 0.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/16-pass@0.5 | gen | 31.88 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/16-pass@0.5/std | gen | 1.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/16-pass@0.75 | gen | 26.09 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/16-pass@0.75/std | gen | 0.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/16-pass@1.0 | gen | 20.29 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/16-pass@1.0/std | gen | 2.05 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/16-pass@0.5 | gen | 37.88 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/16-pass@0.5/std | gen | 1.07 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/16-pass@0.75 | gen | 36.36 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/16-pass@0.75/std | gen | 2.14 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/16-pass@1.0 | gen | 32.58 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/16-pass@1.0/std | gen | 3.21 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/16-pass@0.5 | gen | 43.18 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/16-pass@0.5/std | gen | 3.21 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/16-pass@0.75 | gen | 36.36 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/16-pass@0.75/std | gen | 3.21 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/16-pass@1.0 | gen | 33.33 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/16-pass@1.0/std | gen | 1.07 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/16-pass@0.5 | gen | 16.67 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/16-pass@0.5/std | gen | 0.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/16-pass@0.75 | gen | 16.67 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/16-pass@0.75/std | gen | 0.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/16-pass@1.0 | gen | 12.96 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/16-pass@1.0/std | gen | 2.62 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/16-pass@0.5 | gen | 25.93 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/16-pass@0.5/std | gen | 2.62 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/16-pass@0.75 | gen | 20.37 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/16-pass@0.75/std | gen | 2.62 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/16-pass@1.0 | gen | 9.26 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/16-pass@1.0/std | gen | 5.24 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/16-pass@0.5 | gen | 73.40 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/16-pass@0.5/std | gen | 1.51 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/16-pass@0.75 | gen | 70.93 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/16-pass@0.75/std | gen | 0.85 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/16-pass@1.0 | gen | 68.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/16-pass@1.0/std | gen | 1.13 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass-rate@2 | gen | 63.55 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass-rate@2/std | gen | 1.06 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass@2 | gen | 64.17 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass@2/std | gen | 1.12 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass-rate@2 | gen | 8.33 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass-rate@2/std | gen | 2.68 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass@2 | gen | 9.09 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass@2/std | gen | 3.21 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass-rate@2 | gen | 48.19 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass-rate@2/std | gen | 1.91 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass@2 | gen | 49.28 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass@2/std | gen | 1.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass-rate@2 | gen | 36.59 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass-rate@2/std | gen | 2.05 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass@2 | gen | 37.68 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass@2/std | gen | 1.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass-rate@2 | gen | 45.08 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass-rate@2/std | gen | 1.61 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass@2 | gen | 46.97 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass@2/std | gen | 3.21 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass-rate@2 | gen | 48.86 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass-rate@2/std | gen | 2.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass@2 | gen | 50.76 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass@2/std | gen | 2.14 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass-rate@2 | gen | 22.22 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass-rate@2/std | gen | 0.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass@2 | gen | 22.22 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass@2/std | gen | 0.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass-rate@2 | gen | 36.11 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass-rate@2/std | gen | 2.27 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass@2 | gen | 37.04 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass@2/std | gen | 2.62 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass-rate@2 | gen | 77.70 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass-rate@2/std | gen | 0.61 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass@2 | gen | 78.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass@2/std | gen | 0.66 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass-rate@4 | gen | 62.53 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass-rate@4/std | gen | 0.99 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass@4 | gen | 64.17 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass@4/std | gen | 1.12 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass-rate@4 | gen | 6.63 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass-rate@4/std | gen | 1.51 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass@4 | gen | 9.09 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass@4/std | gen | 3.21 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass-rate@4 | gen | 46.38 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass-rate@4/std | gen | 2.07 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass@4 | gen | 49.28 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass@4/std | gen | 1.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass-rate@4 | gen | 34.96 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass-rate@4/std | gen | 1.47 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass@4 | gen | 37.68 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass@4/std | gen | 1.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass-rate@4 | gen | 43.56 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass-rate@4/std | gen | 1.61 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass@4 | gen | 46.97 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass@4/std | gen | 3.21 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass-rate@4 | gen | 47.16 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass-rate@4/std | gen | 1.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass@4 | gen | 50.76 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass@4/std | gen | 2.14 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass-rate@4 | gen | 20.83 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass-rate@4/std | gen | 1.13 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass@4 | gen | 22.22 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass@4/std | gen | 0.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass-rate@4 | gen | 34.72 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass-rate@4/std | gen | 3.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass@4 | gen | 37.04 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass@4/std | gen | 2.62 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass-rate@4 | gen | 77.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass-rate@4/std | gen | 0.67 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass@4 | gen | 78.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass@4/std | gen | 0.66 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass-rate@8 | gen | 61.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass-rate@8/std | gen | 1.05 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass@8 | gen | 64.17 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass@8/std | gen | 1.12 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass-rate@8 | gen | 5.59 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass-rate@8/std | gen | 0.76 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass@8 | gen | 9.09 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass@8/std | gen | 3.21 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass-rate@8 | gen | 44.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass-rate@8/std | gen | 1.62 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass@8 | gen | 49.28 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass@8/std | gen | 1.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass-rate@8 | gen | 33.42 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass-rate@8/std | gen | 1.25 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass@8 | gen | 37.68 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass@8/std | gen | 1.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass-rate@8 | gen | 41.19 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass-rate@8/std | gen | 1.57 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass@8 | gen | 46.97 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass@8/std | gen | 3.21 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass-rate@8 | gen | 45.55 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass-rate@8/std | gen | 1.57 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass@8 | gen | 50.76 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass@8/std | gen | 2.14 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass-rate@8 | gen | 18.75 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass-rate@8/std | gen | 0.57 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass@8 | gen | 22.22 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass@8/std | gen | 0.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass-rate@8 | gen | 30.79 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass-rate@8/std | gen | 2.82 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass@8 | gen | 37.04 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass@8/std | gen | 2.62 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass-rate@8 | gen | 75.72 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass-rate@8/std | gen | 0.87 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass@8 | gen | 78.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass@8/std | gen | 0.66 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass-rate@16 | gen | 58.20 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass-rate@16/std | gen | 1.11 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass@16 | gen | 64.17 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | pass@16/std | gen | 1.12 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass-rate@16 | gen | 5.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass-rate@16/std | gen | 0.45 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass@16 | gen | 9.09 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AIME2024_en/pass@16/std | gen | 3.21 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass-rate@16 | gen | 39.72 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass-rate@16/std | gen | 1.46 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass@16 | gen | 49.28 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_cn/pass@16/std | gen | 1.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass-rate@16 | gen | 29.39 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass-rate@16/std | gen | 1.22 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass@16 | gen | 37.68 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | AMC_en/pass@16/std | gen | 1.02 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass-rate@16 | gen | 38.12 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass-rate@16/std | gen | 2.04 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass@16 | gen | 46.97 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_cn/pass@16/std | gen | 3.21 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass-rate@16 | gen | 41.19 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass-rate@16/std | gen | 1.87 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass@16 | gen | 50.76 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CCEE_en/pass@16/std | gen | 2.14 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass-rate@16 | gen | 17.36 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass-rate@16/std | gen | 0.57 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass@16 | gen | 22.22 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_cn/pass@16/std | gen | 0.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass-rate@16 | gen | 25.12 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass-rate@16/std | gen | 2.65 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass@16 | gen | 37.04 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | CMO_en/pass@16/std | gen | 2.62 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass-rate@16 | gen | 73.15 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass-rate@16/std | gen | 0.95 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass@16 | gen | 78.00 |
| LiveMathBench_16x3_Skywork-o1-Open-Llama-3.1-8B | caed8f | MATH500_en/pass@16/std | gen | 0.66 |

