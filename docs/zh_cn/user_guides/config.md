# 基于MMEngine的配置文件语法

本页是 OpenCompass 配置语法的速查手册。若你还没有完成过一次评测，请先阅读[使用配置文件完成一次完整评测](config_based_evaluation.md)。

## 基本格式

配置文件是 Python 文件，通过顶层变量声明评测配置细节。最小评测配置包含两个列表：

```python
models = [dict(type=..., abbr='my-model', ...)]
datasets = [dict(type=..., abbr='my-dataset', ...)]
```

常用顶层字段如下：

| 字段         | 作用                                     |
| ------------ | ---------------------------------------- |
| `models`     | 模型后端、路径、生成参数和资源需求       |
| `datasets`   | 数据读取、输入构造、推理方式和评测器     |
| `infer`      | 推理任务的 Partitioner、Runner 与 Task   |
| `eval`       | 评测任务的 Partitioner、Runner 与 Task   |
| `summarizer` | 指标分组、展示顺序与综合分数             |
| `work_dir`   | 实验输出根目录，也可由 `--work-dir` 覆盖 |

## 使用 `read_base()` 复用配置

OpenCompass 使用 MMEngine 的纯 Python 配置继承。导入必须放在 `read_base()` 上下文中：

```python
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import \
        gsm8k_datasets
    from opencompass.configs.models.openai.gpt_6_astra import \
        models as gpt6_models

models = gpt6_models
datasets = gsm8k_datasets
```

从仓库外部配置文件导入时，使用完整的 `opencompass.configs...` 路径最清晰；位于 `opencompass/configs` 内的配置也可以使用相对导入。

## 组合列表

多个模型或数据集可以直接拼接：

```python
models = gpt6_models + other_api_models
datasets = gsm8k_datasets + math_datasets
```

建议在导入时使用有意义的别名，避免多个模块都导出 `models` 或 `datasets` 时发生覆盖。

## 覆盖导入的配置

导入后可以修改列表中的字典。若同一个基础配置还会被其他变量复用，先深拷贝以避免引起误修改：

```python
from copy import deepcopy

models = deepcopy(gpt6_models)
models[0]['query_per_second'] = 2
models[0]['max_workers'] = 16
```

## 配置对象与注册类型

`type` 可以写成导入的 Python 类对象：

```python
from opencompass.summarizers import DefaultSummarizer

summarizer = dict(type=DefaultSummarizer)
```

对于由 MMEngine Registry 管理的组件，`type` 也可以写成注册名称字符串。例如：

```python
infer_cfg = dict(
    prompt_template=dict(
        type='RawPromptTemplate',
        messages=[dict(role='user', content='{question}')],
    ),
    retriever=dict(type='ZeroRetriever'),
    inferencer=dict(type='GenInferencer'),
)
```

构建组件时，上述三个名称会分别在提示词模板、Retriever 和 Inferencer 对应的 Registry 中查找。OpenCompass 的内置 Registry 配置了模块位置，能够在查找时自动导入相应内置模块，因此通常无需再显式导入这些类。

字符串必须与组件在对应 Registry 中的注册名称完全一致；注册时指定了别名的，应填写别名而不是 Python 类名。自定义组件还必须确保其模块已被导入、注册装饰器已执行。不同 Registry 之间不能混用名称，配置中的其余参数也必须与目标组件的构造函数匹配。请注意，并非所有 `type` 字段都覆盖了 Registry 管理。

仓库目前没有统一列出所有 Registry 的 CLI。可以先触发对应 Registry 的内置模块导入，再查看已注册名称。例如，列出全部 Inferencer：

```bash
python -c "from opencompass.registry import ICL_INFERENCERS as R; R.import_from_location(); print('\n'.join(sorted(R.module_dict)))"
```

检查某个名称是否存在时，直接调用 `get()` ：

```bash
python -c "from opencompass.registry import ICL_INFERENCERS as R; print(R.get('GenInferencer'))"
```

根据组件类别，可将查询命令中的 `ICL_INFERENCERS` 替换为 `MODELS`、`LOAD_DATASET`、`RUNNERS`、`PARTITIONERS`、`TASKS`、`ICL_RETRIEVERS`、`ICL_PROMPT_TEMPLATES`、`ICL_EVALUATORS`、`TEXT_POSTPROCESSORS` 或 `DICT_POSTPROCESSORS`。

## 配置与命令行的优先级

配置文件与命令行之间不存在统一的“命令行始终优先”规则。当前入口的部分参数情况按以下方式处理：

- 配置文件中的 `models`、`datasets` 和 `summarizer` 优先生效， CLI 中的 `--models`、`--datasets`、`--summarizer` 以及 Hugging Face 快捷参数不会与其合并。
- `--work-dir` 会覆盖配置中的 `work_dir`；未传入时保留配置值，若配置也未声明则使用 `outputs/default`。
- `--max-num-workers`、`--max-workers-per-gpu` 和 `--retry` 仅在未指定配置文件时生效；其中 `--max-num-workers` 同时设置 Runner 并发数和 `NumWorkerPartitioner.num_worker`。

## 解析和检查配置

使用 MMEngine 单独检查语法：

```bash
python -c "from mmengine.config import Config; Config.fromfile('my_eval.py')"
```

通过 `--dry-run` 检查 OpenCompass 补齐默认项后的配置和任务切分：

```bash
opencompass my_eval.py --dry-run --config-verbose
```
