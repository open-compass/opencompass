# Code Evaluation

This page uses `humaneval` and `mbpp` to explain pass@1 / pass@k configuration. OpenCompass provides built-in evaluators for some code datasets. For multilingual `humaneval-x` workflows that need an independent execution service, use the Docker code-evaluation service to avoid running model-generated code directly in a normal development environment.

## pass@1

To generate one reply and evaluate pass@1, use [opencompass/configs/datasets/humaneval/humaneval_openai_sample_evals_rawprompt_gen_6ce2ca.py](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/humaneval/humaneval_openai_sample_evals_rawprompt_gen_6ce2ca.py) and [opencompass/configs/datasets/mbpp/sanitized_mbpp_mdblock_0shot_nocot_rawprompt_gen_30c1e5.py](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/mbpp/sanitized_mbpp_mdblock_0shot_nocot_rawprompt_gen_30c1e5.py), following the general [Quick Start](../get_started/quick_start.md).

For multilingual evaluation, see [Code Execution Service](#code-execution-service) below.

## pass@k

To generate multiple replies per example for pass@k, repeat the dataset to obtain multiple independent generations. The example below generates 10 replies per problem.

The example below starts from the rawprompt pass@1 configs, reuses their reader and infer configs, and sets `num_repeats=10` to obtain multiple independent generations:

```python
from mmengine.config import read_base

from opencompass.datasets import (
    HumanevalDataset,
    MBPPPassKEvaluator,
    SanitizedMBPPDataset,
)

with read_base():
    from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_rawprompt_gen_6ce2ca import (
        humaneval_eval_cfg,
        humaneval_infer_cfg,
        humaneval_reader_cfg,
    )
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_0shot_nocot_rawprompt_gen_30c1e5 import (
        sanitized_mbpp_infer_cfg,
        sanitized_mbpp_reader_cfg,
    )
    from opencompass.configs.models.openai.gpt_6_astra import models as gpt_6_astra

sanitized_mbpp_passk_reader_cfg = dict(
    sanitized_mbpp_reader_cfg,
    output_column='test_column',
)

sanitized_mbpp_passk_eval_cfg = dict(
    evaluator=dict(type=MBPPPassKEvaluator),
    pred_role='BOT',
)

humaneval_datasets = [
    dict(
        abbr='openai_humaneval_repeat10',
        type=HumanevalDataset,
        path='opencompass/humaneval',
        num_repeats=10,
        reader_cfg=humaneval_reader_cfg,
        infer_cfg=humaneval_infer_cfg,
        eval_cfg=humaneval_eval_cfg,
    )
]

mbpp_datasets = [
    dict(
        abbr='sanitized_mbpp_repeat10',
        type=SanitizedMBPPDataset,
        path='opencompass/sanitized_mbpp',
        num_repeats=10,
        reader_cfg=sanitized_mbpp_passk_reader_cfg,
        infer_cfg=sanitized_mbpp_infer_cfg,
        eval_cfg=sanitized_mbpp_passk_eval_cfg,
    )
]

datasets = humaneval_datasets + mbpp_datasets

models = gpt_6_astra
```

The key field for this approach is `num_repeats` in the dataset config. If you manually convert a normal config, update these fields:

- `num_repeats`: number of dataset repetitions.
- `abbr`: update the dataset abbreviation with the repeat count because dataset size changes; this prevents mismatches with `.cache/dataset_size.json`.

For MBPP / Sanitized MBPP, pass@k requires `eval_cfg.evaluator.type=MBPPPassKEvaluator` and `reader_cfg.output_column='test_column'` so the evaluator can group repeated generations of the same problem by `task_id`.

The model side can directly use the imported GPT-6 config; configure backend-supported sampling parameters in the model config when you need more generation diversity, but do not set `num_return_sequences`.

For a complete example, see [examples/eval_code_passk_repeat_dataset.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_code_passk_repeat_dataset.py).

## Code Execution Service

Some code datasets can be evaluated through an independent service. For installation, deployment, and operation instructions, see the [open-compass/code-evaluator](https://github.com/open-compass/code-evaluator) repository.

### Supported Datasets

#### HumanEval-X

The multilingual [humaneval-x](https://huggingface.co/datasets/THUDM/humaneval-x) dataset. Download the required language file (`xx.jsonl.gz`) from its [download location](https://github.com/THUDM/CodeGeeX2/tree/main/benchmark/humanevalx) and place it under `./data/humanevalx`.

Supported languages are `python`, `cpp`, `go`, `java`, and `js`.

## Local Code Evaluation

When the model inference environment can access the code-evaluation service directly, set `ip_address` and `port` in the dataset evaluator config so OpenCompass can call the service during evaluation.

### Configuration File

We provide [the configuration file](https://github.com/open-compass/opencompass/blob/main/examples/eval_humanevalx_gpt6.py) of using `humanevalx` for evaluation on GPT-6 as reference.

The dataset and related post-processing configuration file can be found at [opencompass/configs/datasets/humanevalx/humanevalx_rawprompt_gen_386eb8.py](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/humanevalx/humanevalx_rawprompt_gen_386eb8.py). Pay attention to the `evaluator` field in `humanevalx_eval_cfg_dict`.

```python
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HumanevalXDataset, HumanevalXEvaluator

humanevalx_reader_cfg = dict(
    input_columns=['prompt'], output_column='declaration', train_split='test')

humanevalx_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(
                role='user',
                content='{prompt}',
            ),
        ]),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024))

humanevalx_eval_cfg_dict = {
    lang: dict(
        evaluator=dict(
            type=HumanevalXEvaluator,
            language=lang,
            ip_address=
            'localhost',  # replace to your code_eval_server ip_address, port
            port=5001),  # refer to https://opencompass.readthedocs.io/en/latest/advanced_guides/code_eval_service.html to launch a server
        pred_role='BOT')
    for lang in ['python', 'cpp', 'go', 'java', 'js']   # do not support rust now
}

# Please download the needed `xx.jsonl.gz` from
# https://github.com/THUDM/CodeGeeX2/tree/main/benchmark/humanevalx
# and move them into `data/humanevalx/` folder
humanevalx_datasets = [
    dict(
        type=HumanevalXDataset,
        abbr=f'humanevalx-{lang}',
        language=lang,
        path='./data/humanevalx',
        reader_cfg=humanevalx_reader_cfg,
        infer_cfg=humanevalx_infer_cfg,
        eval_cfg=humanevalx_eval_cfg_dict[lang])
    for lang in ['python', 'cpp', 'go', 'java', 'js']
]
```

### Task Launch

Refer to the [Five-Minute Quick Start](../get_started/quick_start.md).

## Remote Code Evaluation

Model inference and code evaluation services located in different machines which cannot be accessed directly require prior model inference before collecting the code evaluation results. The configuration file and inference process can be reused from the previous tutorial.

### Collect Inference Results

In OpenCompass's tools folder, there is a script called `collect_code_preds.py` provided to process and collect the inference results after providing the task launch configuration file during startup along with specifying the working directory used corresponding to the task.
It is the same with `-r` option in `run.py`. More details can be referred through the [documentation](https://opencompass.readthedocs.io/en/latest/get_started/quick_start.html#launching-evaluation).

```shell
python tools/collect_code_preds.py [config] [-r latest]
```

The collected results will be organized as following under the `-r` folder:

```
workdir/humanevalx
├── codegeex2-6b
│   ├── humanevalx_cpp.json
│   ├── humanevalx_go.json
│   ├── humanevalx_java.json
│   ├── humanevalx_js.json
│   └── humanevalx_python.json
├── CodeLlama-13b
│   ├── ...
├── CodeLlama-13b-Instruct
│   ├── ...
├── CodeLlama-13b-Python
│   ├── ...
├── ...
```

### Code Evaluation

Make sure your code evaluation service is started, and use `curl` to request:

```shell
curl -X POST -F 'file=@{result_absolute_path}' -F 'dataset={dataset/language}' {your_service_ip_address}:{your_service_port}/evaluate
```

For example:

```shell
curl -X POST -F 'file=@./examples/humanevalx/python.json' -F 'dataset=humanevalx/python' localhost:5000/evaluate
```

The result is:

```
"{\"pass@1\": 37.19512195121951%}"
```

Additionally, we provide a `with-prompt` option, which defaults to `True`. Some models, such as `WizardCoder`, generate complete code and do not need prompt + prediction concatenation. Use the following form in that case:

```shell
curl -X POST -F 'file=@./examples/humanevalx/python.json' -F 'dataset=humanevalx/python' -H 'with-prompt: False' localhost:5000/evaluate
```

## Advanced Tutorial

Besides evaluating the supported code datasets, users might also need:

### Support New Dataset

See [Adding a Dataset](../extension/new_dataset.md).

### Modify Post-Processing

1. For local evaluation, follow the post-processing section in the tutorial on supporting new datasets to modify the post-processing method.
2. For remote evaluation, please modify the post-processing part in the tool's `collect_code_preds.py`.
3. To modify post-processing in the code-evaluation service, see the [open-compass/code-evaluator](https://github.com/open-compass/code-evaluator) repository.
