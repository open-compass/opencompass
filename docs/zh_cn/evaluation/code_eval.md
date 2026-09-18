# 代码评测

这里以 `humaneval` 和 `mbpp` 为例介绍 pass@1 / pass@k 的配置方式。OpenCompass 对部分代码数据集提供内置 Evaluator；对于需要独立执行服务的多语言 `humaneval-x`，可以通过 Docker 代码评测服务完成评测，避免在普通开发环境中直接执行模型生成的代码。

## pass@1

如果只需要生成单条回复来评测 pass@1，可以直接使用 [opencompass/configs/datasets/humaneval/humaneval_openai_sample_evals_rawprompt_gen_6ce2ca.py](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/humaneval/humaneval_openai_sample_evals_rawprompt_gen_6ce2ca.py) 和 [opencompass/configs/datasets/mbpp/sanitized_mbpp_mdblock_0shot_nocot_rawprompt_gen_30c1e5.py](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/mbpp/sanitized_mbpp_mdblock_0shot_nocot_rawprompt_gen_30c1e5.py)，并参考通用的[快速上手教程](../get_started/quick_start.md)即可。

如果要进行多语言评测，可以参考本文[代码执行服务](#代码执行服务)一节。

## pass@k

如果需要对单个样本生成多条回复来评测 pass@k，可以通过重复构造数据集来得到多次独立生成结果。这里以每题 10 条回复为例：

下面以 pass@1 的 rawprompt 配置为基础，复用其 reader 和 infer 配置，并通过 `num_repeats=10` 重复构造数据集来得到多次独立生成结果：

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

这种方式的关键是数据集配置中的 `num_repeats`。如果需要从普通配置手动改，需要同步修改以下字段：

- `num_repeats`：数据集重复次数。
- `abbr`：数据集缩写建议随重复次数一并修改，因为数据集数量会发生变化，可避免与 `.cache/dataset_size.json` 中的数值不一致。

对于 MBPP / Sanitized MBPP，pass@k 需要把 `eval_cfg.evaluator.type` 设为 `MBPPPassKEvaluator`，并把 `reader_cfg.output_column` 改为 `test_column`，以便 Evaluator 按 `task_id` 聚合同一道题的多次生成结果。

模型侧可以直接使用导入的 GPT-6 配置；如需控制生成多样性，应在模型配置中设置对应后端支持的采样参数，但不需要设置 `num_return_sequences`。

具体可以参考以下配置文件：[examples/eval_code_passk_repeat_dataset.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_code_passk_repeat_dataset.py)。

## 代码执行服务

部分代码数据集可以通过独立服务执行评测。代码评测服务的安装、部署和运行方式请参考 [open-compass/code-evaluator](https://github.com/open-compass/code-evaluator) 仓库。

### 支持的数据集

#### HumanEval-X

多编程语言的数据集 [humaneval-x](https://huggingface.co/datasets/THUDM/humaneval-x)。数据集[下载地址](https://github.com/THUDM/CodeGeeX2/tree/main/benchmark/humanevalx)，请下载需要评测的语言（`xx.jsonl.gz`）文件，并放入 `./data/humanevalx` 文件夹。

目前支持的语言有 `python`、`cpp`、`go`、`java`、`js`。

## 本地代码评测

模型推理环境可以直接访问代码评测服务时，可以在数据集配置的 evaluator 中设置 `ip_address` 和 `port`，由 OpenCompass 在评测阶段直接调用服务。

### 配置文件

我们已经提供了 humaneval-x 在 GPT-6 上评估的[配置文件](https://github.com/open-compass/opencompass/blob/main/examples/eval_humanevalx_gpt6.py)作为参考。
其中数据集以及相关后处理的配置文件可以参考 [opencompass/configs/datasets/humanevalx/humanevalx_rawprompt_gen_386eb8.py](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/humanevalx/humanevalx_rawprompt_gen_386eb8.py)，需要注意 `humanevalx_eval_cfg_dict` 中的 `evaluator` 字段。

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

### 任务启动

参考[五分钟快速开始](../get_started/quick_start.md)

## 异地代码评测

模型推理和代码评测服务分别在不可访问的不同机器中，需要先进行模型推理，收集代码推理结果。配置文件和推理流程都可以复用上面的教程。

### 收集推理结果

OpenCompass 在 `tools` 中提供了 `collect_code_preds.py` 脚本对推理结果进行后处理并收集，我们只需要提供启动任务时的配置文件，以及指定复用对应任务的工作目录，其参数含义与 `opencompass` 的 `--reuse` 一致，细节可参考[文档](https://opencompass.readthedocs.io/zh-cn/latest/get_started/quick_start.html#id4)。

```shell
python tools/collect_code_preds.py [config] [-r latest]
```

收集到的结果将会按照以下的目录结构保存到 `-r` 对应的工作目录中：

```
workdir/humanevalx
├── codegeex2-6b
│   ├── humanevalx_cpp.json
│   ├── humanevalx_go.json
│   ├── humanevalx_java.json
│   ├── humanevalx_js.json
│   └── humanevalx_python.json
├── CodeLlama-13b
│   ├── ...
├── CodeLlama-13b-Instruct
│   ├── ...
├── CodeLlama-13b-Python
│   ├── ...
├── ...
```

### 代码评测

确保代码评测服务启动的情况下，使用 `curl` 提交请求：

```shell
curl -X POST -F 'file=@{result_absolute_path}' -F 'dataset={dataset/language}' {your_service_ip_address}:{your_service_port}/evaluate
```

例如：

```shell
curl -X POST -F 'file=@./examples/humanevalx/python.json' -F 'dataset=humanevalx/python' localhost:5000/evaluate
```

得到结果：

```
"{\"pass@1\": 37.19512195121951%}"
```

另外我们额外提供了 `with-prompt` 选项（默认为 `True`），由于有些模型生成结果包含完整代码（如 WizardCoder），不需要按 prompt + prediction 的形式拼接，可以参考以下命令进行评测。

```shell
curl -X POST -F 'file=@./examples/humanevalx/python.json' -F 'dataset=humanevalx/python' -H 'with-prompt: False' localhost:5000/evaluate
```

## 进阶教程

除了评测已支持的代码数据集以外，用户还可能有以下需求：

### 支持新数据集

可以参考[支持新数据集](../extension/new_dataset.md)教程

### 修改后处理

1. 本地评测中，可以按照支持新数据集教程中的后处理部分来修改后处理方法；
2. 异地评测中，可以修改 `tools/collect_code_preds.py` 中的后处理部分；
3. 如需修改代码评测服务中的后处理，请参考 [open-compass/code-evaluator](https://github.com/open-compass/code-evaluator) 仓库。
