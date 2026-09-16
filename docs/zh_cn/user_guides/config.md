# 配置语法与复用参考

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

`type` 可以写导入的 Python 类：

```python
from opencompass.summarizers import DefaultSummarizer

summarizer = dict(type=DefaultSummarizer)
```

OpenCompass 解析后会通过注册表构建对应组件。配置中的参数必须与组件构造函数匹配；模型、数据集、Runner 等不同组件不能交换字段。

## 配置与命令行的优先级

一般情况下，配置文件声明实验，命令行控制本次启动。常见覆盖关系包括：

- `--work-dir` 覆盖配置的 `work_dir`；
- `--debug` 打开 Runner 的调试模式；
- `--max-num-workers` 只在 CLI 自动生成默认 Runner 时生效，显式配置的同名字段优先；
- `--mode`、`--reuse` 控制执行阶段与产物复用。

运行 `opencompass --help` 查看当前版本的完整参数，不要把旧版本文档中的参数直接复制到新环境。

## 解析和检查配置

使用 MMEngine 单独检查语法：

```bash
python -c "from mmengine.config import Config; Config.fromfile('my_eval.py')"
```

通过 `--dry-run` 检查 OpenCompass 补齐默认项后的配置和任务切分：

```bash
opencompass my_eval.py --dry-run --config-verbose
```
