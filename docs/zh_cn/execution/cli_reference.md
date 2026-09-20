# 命令行参数参考

`opencompass` 命令的基本格式如下：

```bash
opencompass [config] [options]
```

本页以当前版本 `opencompass/cli/main.py` 的实现为准。若代码更新导致参数发生变化，请以以下命令的输出为最终依据：

```bash
opencompass --help
```

## 配置入口与配置查找

| 参数                               | 默认值    | 说明                                                                                                                                                     |
| ---------------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config`                           | 无        | 可选的配置文件参数，指定 Python 配置文件路径。                                                                                                           |
| `-h`, `--help`                     | —         | 显示帮助信息。                                                                                                                                           |
| `--models MODEL [MODEL ...]`       | 无        | 按名称查找并加载一个或多个模型配置。                                                                                                                     |
| `--datasets DATASET [DATASET ...]` | 无        | 按名称查找并加载一个或多个数据集配置。                                                                                                                   |
| `--summarizer SUMMARIZER`          | `example` | 在快捷配置模式下选择结果汇总配置；支持使用 `文件名/配置键` 指定配置对象。                                                                                |
| `--config-dir DIR`                 | `configs` | 指定自定义配置根目录；OpenCompass 会在其 `models/`、`datasets/`、`dataset_collections/` 和 `summarizers/` 子目录中查找配置，同时保留内置配置的搜索路径。 |

如果指定了 `config`，OpenCompass 会优先读取该文件，`--models`、`--datasets`、`--summarizer`、`--hf-*` 和 `--custom-dataset-*` 等快捷构造参数不会用于替换其中的模型或数据集配置。未提供配置文件时，可以采用以下任一入口：

- 使用 `--models` 与 `--datasets` 加载已有配置；
- 使用 `--hf-path` 快速构造 Hugging Face 模型，并通过 `--datasets` 指定数据集；
- 使用 `--models` 或 `--hf-path` 指定模型，并通过 `--custom-dataset-path` 构造自定义数据集。

## 执行阶段与工作目录

| 参数                                  | 默认值            | 说明                                                                                                                         |
| ------------------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `-m`, `--mode {all,infer,eval,viz}`   | `all`             | 选择执行阶段：`all` 执行完整流程，`infer` 仅推理，`eval` 仅评分，`viz` 仅汇总。                                              |
| `-r`, `--reuse [TIMESTAMP]`           | 无                | 复用指定时间戳的实验目录；省略时间戳时使用工作目录下按名称排序的最新目录。                                                   |
| `-w`, `--work-dir DIR`                | `outputs/default` | 指定实验输出根目录，实际产物保存在其时间戳子目录中。                                                                         |
| `--debug`                             | `False`           | 启用调试模式，使 Runner 顺序执行任务并直接输出日志，适合定位首次运行问题。                                                   |
| `--dry-run`                           | `False`           | 解析配置并执行任务划分，但不启动推理或评测；启用时会同时打开调试日志级别。                                                   |
| `-a`, `--accelerator {vllm,lmdeploy}` | 无                | 尝试在一站式部署与评测中将支持的 Hugging Face 本地模型配置转换为 vLLM 或 LMDeploy 配置；不支持的模型类型保持不变并输出警告。 |
| `--config-verbose`                    | `False`           | 输出当前加载并处理后的实验配置。                                                                                             |
| `-l`, `--lark`                        | `False`           | 启用飞书机器人任务通知，需要配置中同时提供 `lark_bot_url`。                                                                  |

## 任务划分与 Runner

| 参数                      | 默认值  | 说明                                                                                                                                                                     |
| ------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--max-num-workers N`     | `1`     | 为入口自动补齐的推理或评测配置设置默认 Runner 的最大并发任务数，并设置默认推理 Partitioner 的 `num_worker`。显式配置相应阶段时通常不覆盖已有值。                         |
| `--max-workers-per-gpu N` | `1`     | 设置自动生成的 `LocalRunner` 在每张 GPU 上可同时运行的最大任务数。                                                                                                       |
| `--slurm -p PARTITION`    | `False` | 强制使用 `SlurmRunner`，并覆盖已有的 `infer` 和 `eval` 执行配置。`-p/--partition` 为必填参数；还可通过 `-q/--quotatype`、`--qos` 和 `--retry`（默认 2）设置配额类型、Quality of Service 与失败重试次数。该参数与 `--dlc` 互斥。 |
| `--dlc --aliyun-cfg PATH` | `False` | 强制使用阿里云 PAI-DLC Runner，并覆盖已有的 `infer` 和 `eval` 执行配置。`--aliyun-cfg` 指定 DLC 配置文件，默认值为 `~/.aliyun.cfg`，路径必须存在；`--retry`（默认 2）设置失败重试次数。该参数与 `--slurm` 互斥。 |

## 推理、评测与分析输出

| 参数                           | 默认值  | 说明                                                                                                    |
| ------------------------------ | ------- | ------------------------------------------------------------------------------------------------------- |
| `--dump-eval-details [BOOL]`   | `True`  | 保存逐样本评测明细。使用 `--dump-eval-details False` 可以关闭；省略参数值时仍为 `True`。                |
| `--dump-res-length`            | `False` | 向推理任务传入响应长度统计开关；是否支持取决于所用 Inferencer。                                         |
| `--dump-only-message-path DIR` | 无      | 仅导出构造后的消息而不执行模型请求，当前仅支持 `GenInferencer`。                                        |
| `--dump-extract-rate`          | `False` | 要求评测任务计算并保存答案抽取率。                                                                      |
| `--analysis-repeat`            | `False` | 在汇总阶段分析重复预测，并写入重复输出分析文件。                                                        |
| `--dataset-num-runs N`         | `1`     | 在 CLI 快捷配置模式下，将加载的数据集配置中的 `n` 和 `k` 统一设为 `N`；显式传入配置文件时不应用此参数。 |

## 快速构造 Hugging Face 模型

以下参数仅用于未通过 `config` 或 `--models` 提供模型时，根据 `--hf-path` 快速生成模型配置：

| 参数                                            | 默认值         | 说明                                                       |
| ----------------------------------------------- | -------------- | ---------------------------------------------------------- |
| `--hf-type {base,chat}`                         | `chat`         | 选择基础模型或对话模型封装。                               |
| `--hf-path PATH`                                | 无             | 指定 Hugging Face 模型路径或仓库 ID。                      |
| `--model-kwargs KEY=VALUE [KEY=VALUE ...]`      | `{}`           | 传递模型加载参数。                                         |
| `--tokenizer-path PATH`                         | 与模型路径一致 | 指定 tokenizer 路径或仓库 ID。                             |
| `--tokenizer-kwargs KEY=VALUE [KEY=VALUE ...]`  | `{}`           | 传递 tokenizer 加载参数。                                  |
| `--peft-path PATH`                              | 无             | 指定 PEFT 权重路径。                                       |
| `--peft-kwargs KEY=VALUE [KEY=VALUE ...]`       | `{}`           | 传递 PEFT 加载参数。                                       |
| `--generation-kwargs KEY=VALUE [KEY=VALUE ...]` | `{}`           | 传递生成参数。                                             |
| `--max-seq-len N`                               | 无             | 设置模型支持的最大序列长度。                               |
| `--max-out-len N`                               | `256`          | 设置最大输出长度。                                         |
| `--min-out-len N`                               | `1`            | 设置最小输出长度。                                         |
| `--batch-size N`                                | `8`            | 设置推理批大小。                                           |
| `--hf-num-gpus N`                               | `1`            | 设置 Hugging Face 模型任务使用的 GPU 数量。                |
| `--pad-token-id N`                              | 无             | 指定 padding token ID。                                    |
| `--stop-words WORD [WORD ...]`                  | 空列表         | 指定一个或多个停止词。                                     |
| `--num-gpus N`                                  | —              | 已废弃；当前版本传入该参数会报错，应改用 `--hf-num-gpus`。 |

## 快速构造自定义数据集

以下参数用于从本地文件快速生成数据集配置：

| 参数                                      | 默认值   | 说明                                                                 |
| ----------------------------------------- | -------- | -------------------------------------------------------------------- |
| `--custom-dataset-path PATH`              | 无       | 指定自定义数据集文件。未提供 `config` 或 `--datasets` 时为必填参数。 |
| `--custom-dataset-meta-path PATH`         | 无       | 指定自定义数据集的元信息文件。                                       |
| `--custom-dataset-data-type {mcq,qa}`     | 自动判断 | 指定数据类型：选择题或问答题。                                       |
| `--custom-dataset-infer-method {gen,ppl}` | 自动判断 | 指定生成式或 PPL 推理方式。                                          |

## 结果持久化参数

| 参数                        | 默认值  | 说明                                                                                               |
| --------------------------- | ------- | -------------------------------------------------------------------------------------------------- |
| `-sp`, `--station-path DIR` | 无      | 指定结果共享目录。传入该参数或在配置中设置 `station_path` 后，运行结束时会将结果保存到共享目录。   |
| `--read-from-station`       | `False` | 运行前从共享目录读取已有结果，将其写入当前实验的 `results/`，并跳过已经存在结果的模型—数据集组合。 |
| `--station-overwrite`       | `False` | 向共享目录保存结果时允许覆盖已有文件。                                                             |
