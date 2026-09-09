# OpenCompass 工作流与核心概念

OpenCompass 评测能够拉起一条可以拆分、并行、恢复和复用的流水线：

```text
实验配置
  ├─ models：被评测模型
  ├─ datasets：样本、输入构造与评测方法
  └─ infer / eval / summarizer：任务编排与结果展示
          ↓
配置解析与任务划分
          ↓
推理（结果写入predictions）
          ↓
评测（结果写入results）
          ↓
汇总（结果写入summary）
```

## 评测配置

配置是一次实验的唯一入口。最小配置只需要 `models` 和 `datasets`；OpenCompass 会补齐本地运行所需的默认推理与评测任务。正式实验通常还会指定 `work_dir`、`infer`、`eval` 或 `summarizer`，以固定并发方式和汇总口径。

配置文件使用 Python 语法，并通过 MMEngine 的 `read_base()` 复用仓库中的模型、数据集和汇总配置。配置不是随意执行的启动脚本：其职责是声明实验对象和策略，实际调度由 `opencompass` 命令完成。

## 模型

模型配置描述“如何调用模型”和“运行一个实例需要多少资源”，包括模型后端、权重或接口地址、上下文长度、最大输出长度、批大小、生成参数以及 `run_cfg`。本地 Hugging Face 权重、OpenAI 兼容接口、LMDeploy、vLLM 和多模态模型会使用不同模型类。

详见[模型接入](models.md)。

## 数据集

OpenCompass 中的数据集配置不只包含数据路径。它通常同时声明：

- `reader_cfg`：输入字段、答案字段和数据切分；
- `infer_cfg`：提示词、样例检索器和生成或 PPL 推理器；
- `eval_cfg`：答案后处理和指标计算方式。

因此，同一原始数据可能有多个配置变体，例如不同的 few-shot、提示词或评测器。详见[数据集配置](datasets.md)。

## 推理、评测与汇总

推理阶段由 Partitioner 把“模型 × 数据集”拆成任务，Runner 决定任务在本地、或其他的集群环境中如何执行，Task 完成实际推理任务。输出写入 `predictions/`。

评测阶段读取预测结果，由数据集的 Evaluator 计算分数并写入 `results/`。Summarizer 再把各子集结果整理为终端表格和汇总文件。推理结果与评测结果分开保存，因此可以通过 `--reuse` 只重做缺失阶段，或使用 `--mode eval`、`--mode viz` 处理已有结果。

## 工作目录与可复现性

未指定时，输出位于 `outputs/default/<时间戳>/`。建议每个实验显式指定稳定的 `--work-dir`，并保留其中的 `configs/` 配置快照。模型版本、数据版本、依赖环境、随机参数和 Judge 模型都会影响结果，不能只记录最终分数。

上述内容仅对opencompass工作流的各个步骤做粗略的描述。想要完整启动一次基于配置文件的评测，请参考[使用配置文件完成一次完整评测](config_based_evaluation.md)。
