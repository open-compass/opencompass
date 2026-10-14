# 数据集选择与配置

OpenCompass 的一份数据集配置同时定义数据读取、模型输入和评分规则。同一数据集可以有多个配置变体，不同评测方案得到的结果可能存在差异。

## 选择配置

```bash
python tools/list_configs.py mmlu gsm8k  # 查找与 mmlu、gsm8k 相关的配置
```

数据集配置文件通常位于 `opencompass/configs/datasets/<数据集>/`。文件名通常包含 `gen`、`ppl`、`rawprompt`、few-shot 数量和哈希等标识，用于区分不同评测方案。

选择配置时请确认以下信息：

- 数据来源、版本、split 和样本范围；
- 输入字段、答案字段和可能存在的多模态输入；
- Prompt 类型、few-shot 数量和推理方式；
- Evaluator 和后处理规则，以及是否依赖 Judge 模型或外部评测服务。

## 数据集配置结构

一份数据集配置由数据加载参数以及 `reader_cfg`、`infer_cfg`、`eval_cfg` 三部分组成：

```python
datasets = [
    dict(
        type=MyDataset,
        abbr='my-dataset',
        path='data/or/hub-id',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]
```

- `type`：注册到 OpenCompass 的数据集类，负责将原始数据加载为 Hugging Face `Dataset` 或 `DatasetDict`。
- `abbr`：该配置在任务目录和汇总结果中的简称。同一原始数据的不同评测配置应使用可区分的简称。
- `path`：数据集路径或仓库标识。不同数据集类还可能接收 `name`、`split`、`task` 等指定子集的参数。
- `reader_cfg`：指定参与评测的字段、数据划分和样本范围。
- `infer_cfg`：指定提示词构造、few-shot 样本检索和推理方式。
- `eval_cfg`：指定推理结果的后处理方式和评分规则。

### `reader_cfg`：读取字段与数据划分

`reader_cfg` 的基本格式如下：

```python
reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='train',
    test_split='test',
    train_range=None,
    test_range='[:100]',
)
```

各字段的含义如下：

- `input_columns`：构造模型输入时使用的字段列表，例如题目、选项或上下文。
- `output_column`：参考答案所在字段；无需参考答案的任务可以设为 `None`。
- `train_split`、`test_split`：分别指定 Retriever 选取 few-shot 示例的数据划分，以及实际执行推理和评分的数据划分，默认值为 `train` 和 `test`。如果数据只有一个划分，则无需指定此参数。
- `train_range`、`test_range`：限制对应划分的样本范围。`None` 表示使用全部样本；整数表示从打乱后的数据中选取固定数量；`0` 到 `1` 之间的浮点数表示抽取相应比例；切片字符串（如 `'[:100]'`、`'[100:200]'`）表示按原始顺序选择指定区间。全量评测时可以省略这两个字段。

配置完成后，应确认 `input_columns`、`output_column` 以及提示词中的占位符都能在加载后的数据中找到。需要快速试跑固定的前若干条数据时，应使用 `test_range='[:N]'`。

### `infer_cfg`：提示词、检索与推理方式

生成式评测最常见的格式如下：

```python
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(role='user', content='{question}\n请给出答案。'),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

`infer_cfg` 包含以下常用字段：

- `prompt_template`：通过提示词模板将数据集样本转换为模型输入。
- `retriever`：决定从训练划分中选择哪些 few-shot 示例。`ZeroRetriever` 不从数据集中检索示例；`FixKRetriever`、`RandomRetriever` 等则按相应策略选择示例。
- `inferencer`：决定推理方式。`GenInferencer` 让模型直接生成答案，并可设置 `max_out_len`、`stopping_criteria` 等推理参数；若在此处显式设置 `max_out_len`，其优先级高于模型配置中的默认值。

提示词占位符、对话消息和 few-shot 插入方式详见[提示词模板](../prompt/raw_prompt_template.md)；修改模板后，可通过[提示词预览与调试](../prompt/debugging.md)检查最终输入。

PPL 评测的常用结构如下：

```python
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'yes': '{question} yes',
            'no': '{question} no',
        },
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)
```

候选模板的键应覆盖 `output_column` 中的标签值，也可以通过 `PPLInferencer` 的 `labels` 参数显式指定候选标签。PPL 评测要求模型后端支持对输入计算对数似然，并非所有 API 模型都具备该能力。

### `eval_cfg`：后处理与评分

当模型输出不能直接与参考答案比较时，可以分别进行后处理。例如：

```python
from opencompass.datasets import (Gsm8kEvaluator,
                                  gsm8k_dataset_postprocess,
                                  gsm8k_postprocess)

eval_cfg = dict(
    evaluator=dict(type=Gsm8kEvaluator),
    pred_postprocessor=dict(type=gsm8k_postprocess),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
)
```

各字段的作用如下：

- `evaluator`：评分器配置。`type` 指定具体 Evaluator，其余字段作为初始化参数传入。常见评分方式包括准确率、精确匹配、数学答案校验、代码执行和模型裁判。
- `pred_postprocessor`：在评分前处理模型预测，例如提取选项字母、数字或特定标签中的答案。
- `dataset_postprocessor`：在评分前处理 `output_column` 中的参考答案，使其格式与处理后的预测一致。
- `pred_role`：从本地对话模型的输出中提取指定角色内容，仅在模型配置了相应 `meta_template` 时按需使用。

`evaluator type` 是基本评分流程的必要项，其他字段均为可选项。

数据的缓存和离线规则详见[数据来源、缓存与离线运行](data_and_cache.md)。数据集的完整自定义方法参阅[新增数据集](../extension/new_dataset.md)。

## 数据集统计

下表根据仓库根目录的 `dataset-index.yml` 自动生成，列出 OpenCompass 已登记的数据集、类别、资源地址及推荐配置，支持模糊搜索。

```{include} ../dataset_statistics.inc
```
