# LiveMathBench

## v202412

### Details of Datsets

| dataset | language | #single-choice | #multiple-choice | #fill-in-the-blank | #problem-solving |
| -- | -- | -- | -- | -- | -- |
| AMC | cn | 0 | 0 | 0 | 46 |
| AMC | en | 0 | 0 | 0 | 46 |
| CCEE | cn | 0 | 0 | 13 | 31 |
| CCEE | en | 0 | 0 | 13 | 31 |
| CNMO | cn | 0 | 0 | 0 | 18 |
| CNMO | en | 0 | 0 | 0 | 18 |
| WLPMC | cn | 0 | 0 | 0 | 11 |
| WLPMC | en | 0 | 0 | 0 | 11 |


### How to use

#### G-Pass@k
```python
from mmengine.config import read_base

with read_base():
    from opencompass.datasets.livemathbench_gen import livemathbench_datasets

livemathbench_datasets[0]['eval_cfg']['evaluator'].update(
    {
        'model_name': 'Qwen/Qwen2.5-72B-Instruct', 
        'url': [
            'http://0.0.0.0:23333/v1', 
            '...'
        ]  # set url of evaluation models
    }
)
livemathbench_dataset['infer_cfg']['inferencer'].update(dict(
    max_out_len=32768 # for o1-like models you need to update max_out_len
))

```

#### Greedy
```python
from mmengine.config import read_base

with read_base():
    from opencompass.datasets.livemathbench_greedy_gen import livemathbench_datasets

livemathbench_datasets[0]['eval_cfg']['evaluator'].update(
    {
        'model_name': 'Qwen/Qwen2.5-72B-Instruct', 
        'url': [
            'http://0.0.0.0:23333/v1', 
            '...'
        ]  # set url of evaluation models
    }
)
livemathbench_dataset['infer_cfg']['inferencer'].update(dict(
    max_out_len=32768 # for o1-like models you need to update max_out_len
))

```

### Output Samples

| dataset | version | metric | mode | Qwen2.5-72B-Instruct |
|----- | ----- | ----- | ----- | -----|
| LiveMathBench | 9befbf | G-Pass@16_0.0 | gen | xx.xx |
| LiveMathBench | caed8f | G-Pass@16_0.25 | gen | xx.xx |
| LiveMathBench | caed8f | G-Pass@16_0.5 | gen | xx.xx |
| LiveMathBench | caed8f | G-Pass@16_0.75 | gen | xx.xx |
| LiveMathBench | caed8f | G-Pass@16_1.0 | gen | xx.xx |

