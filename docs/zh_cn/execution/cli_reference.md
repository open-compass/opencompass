# 命令行参数参考

当前版本的最终依据是：

```bash
opencompass --help
```

## 实验入口

| 参数 | 含义 |
| --- | --- |
| `config` | Python 实验配置文件，可省略 |
| `--models` | 从配置目录按名称选择模型 |
| `--datasets` | 从配置目录按名称选择数据集 |
| `--summarizer` | 按名称选择汇总配置 |
| `--config-dir` | 额外的配置搜索目录 |

未提供配置文件时，必须使用 `--models` 加 `--datasets`，或使用 `--hf-path` 构造一个 Hugging Face 模型并指定数据集。

## 执行和输出

| 参数 | 含义 |
| --- | --- |
| `--dry-run` | 解析配置并切分任务，不执行推理 |
| `--debug` | 单进程执行并在终端显示日志 |
| `--mode {all,infer,eval,viz}` | 选择执行阶段 |
| `--reuse [TIMESTAMP]` | 复用指定或最新时间戳目录 |
| `--work-dir` | 输出根目录 |
| `--config-verbose` | 打印最终配置 |
| `--dump-eval-details False` | 关闭默认开启的逐样本评测明细 |
| `--dump-res-length` | 记录回复长度 |
| `--analysis-repeat` | 在汇总阶段分析重复预测 |
| `--dump-extract-rate` | 输出答案抽取率 |

## 并发和后端

| 参数 | 含义 |
| --- | --- |
| `--max-num-workers` | 默认 Runner 的最大并发任务数 |
| `--max-workers-per-gpu` | LocalRunner 每张 GPU 的最大任务数 |
| `--slurm -p PARTITION` | 使用 Slurm |
| `--dlc --aliyun-cfg PATH` | 使用 DLC |
| `--accelerator {vllm,lmdeploy}` | 尝试转换受支持的 HF 模型配置 |
| `--retry` | Slurm/DLC 默认 Runner 的失败重试次数 |

## 快速构造 Hugging Face 模型

常用参数包括 `--hf-type`、`--hf-path`、`--tokenizer-path`、`--model-kwargs`、`--tokenizer-kwargs`、`--generation-kwargs`、`--max-seq-len`、`--max-out-len`、`--batch-size` 和 `--hf-num-gpus`。复杂或需要复现的模型设置应写入配置文件。

`--dataset-num-runs N` 可复制每个数据集配置执行 N 次。使用随机生成时应记录每次结果和聚合方法。
