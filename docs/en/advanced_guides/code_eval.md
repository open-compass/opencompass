# Code Evaluation Tutorial

This tutorial primarily focuses on evaluating a model's coding proficiency, using `humaneval` and `mbpp` as examples.

## pass@1

If you only need to generate a single response to evaluate the pass@1 performance, you can directly use [configs/datasets/humaneval/humaneval_gen_8e312c.py](https://github.com/open-compass/opencompass/blob/main/configs/datasets/humaneval/humaneval_gen_8e312c.py) and [configs/datasets/mbpp/deprecated_mbpp_gen_1e1056.py](https://github.com/open-compass/opencompass/blob/main/configs/datasets/mbpp/deprecated_mbpp_gen_1e1056.py), referring to the general [quick start tutorial](../get_started/quick_start.md).

For multilingual evaluation, please refer to the [Multilingual Code Evaluation Tutorial](./code_eval_service.md).

## pass@k

If you need to generate multiple responses for a single example to evaluate the pass@k performance, consider the following two situations. Here we take 10 responses as an example:

### Typical Situation

For most models that support the `num_return_sequences` parameter in HF's generation, we can use it directly to obtain multiple responses. Refer to the following configuration file:

```python
from opencompass.datasets import MBPPDatasetV2, MBPPPassKEvaluator

with read_base():
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from .datasets.mbpp.deprecated_mbpp_gen_1e1056 import mbpp_datasets

mbpp_datasets[0]['type'] = MBPPDatasetV2
mbpp_datasets[0]['eval_cfg']['evaluator']['type'] = MBPPPassKEvaluator
mbpp_datasets[0]['reader_cfg']['output_column'] = 'test_column'

datasets = []
datasets += humaneval_datasets
datasets += mbpp_datasets

models = [
    dict(
        type=HuggingFaceCausalLM,
        ...,
        generation_kwargs=dict(
            num_return_sequences=10,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        ),
        ...,
    )
]
```

For `mbpp`, new changes are needed in the dataset and evaluation, so we simultaneously modify the `type`, `eval_cfg.evaluator.type`, `reader_cfg.output_column` fields to accommodate these requirements.

We also need model responses with randomness, thus setting the `generation_kwargs` parameter is necessary. Note that we need to set `num_return_sequences` to get the number of responses.

Note: `num_return_sequences` must be greater than or equal to k, as pass@k itself is a probability estimate.

You can specifically refer to the following configuration file [configs/eval_code_passk.py](https://github.com/open-compass/opencompass/blob/main/configs/eval_code_passk.py)

### For Models That Do Not Support Multiple Responses

This applies to some HF models with poorly designed APIs or missing features. In this case, we need to repeatedly construct datasets to achieve multiple response effects. Refer to the following configuration:

```python
from opencompass.datasets import MBPPDatasetV2, MBPPPassKEvaluator

with read_base():
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from .datasets.mbpp.deprecated_mbpp_gen_1e1056 import mbpp_datasets

humaneval_datasets[0]['abbr'] = 'openai_humaneval_pass10'
humaneval_datasets[0]['num_repeats'] = 10
mbpp_datasets[0]['abbr'] = 'mbpp_pass10'
mbpp_datasets[0]['num_repeats'] = 10
mbpp_datasets[0]['type'] = MBPPDatasetV2
mbpp_datasets[0]['eval_cfg']['evaluator']['type'] = MBPPPassKEvaluator
mbpp_datasets[0]['reader_cfg']['output_column'] = 'test_column'

datasets = []
datasets += humaneval_datasets
datasets += mbpp_datasets

models = [
    dict(
        type=HuggingFaceCausalLM,
        ...,
        generation_kwargs=dict(
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        ),
        ...,
    )
]
```

Since the dataset's prompt has not been modified, we need to replace the corresponding fields to achieve the purpose of repeating the dataset.
You need to modify these fields:

- `num_repeats`: the number of times the dataset is repeated
- `abbr`: It's best to modify the dataset abbreviation along with the number of repetitions because the number of datasets will change, preventing potential issues arising from discrepancies with the values in `.cache/dataset_size.json`.

For `mbpp`, modify the `type`, `eval_cfg.evaluator.type`, `reader_cfg.output_column` fields as well.

We also need model responses with randomness, thus setting the `generation_kwargs` parameter is necessary.

You can specifically refer to the following configuration file [configs/eval_code_passk_repeat_dataset.py](https://github.com/open-compass/opencompass/blob/main/configs/eval_code_passk_repeat_dataset.py)
