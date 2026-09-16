# OpenCompass 工作流与核心概念

OpenCompass 评测能够拉起一条可以拆分、并行、恢复和复用的流水线：

```text
实验配置
  ├─ models：被评测模型
  ├─ datasets：样本、输入构造与评测方法
  └─ infer / eval / summarizer：任务执行策略与结果展示
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

配置是一次实验的唯一入口。最小配置只需要 `models` 和 `datasets`；OpenCompass 会补齐本地运行所需的默认推理与评测任务。正式实验通常还会指定 `work_dir`、`infer`、`eval` 或 `summarizer`，以指定并发方式和结果汇总格式。

配置文件使用 Python 语法，并通过 MMEngine 的 `read_base()` 复用仓库中的模型、数据集和结果汇总的配置。配置不是随意执行的启动脚本：其职责是声明评测对象和评测策略，实际调度由 `opencompass` 命令完成。

## 模型

模型配置描述“如何调用模型”和“运行所需的资源”，包括模型后端、权重或接口地址、上下文长度、最大输出长度、并发数、模型超参数等。常见入口包括 OpenAI 兼容接口、各厂商 SDK，以及本地 Hugging Face、LMDeploy、vLLM 和多模态模型类。

详见[模型接入](models.md)。

## 数据集

OpenCompass 中的数据集配置不只包含数据路径。它通常同时声明：

- `reader_cfg`：输入字段、答案字段和数据切分策略；
- `infer_cfg`：提示词结构、few-shot 配置、PPL / Gen 等推理方式；
- `eval_cfg`：对模型输出的后处理和指标计算方式。

因此，同一原始数据可能有多个配置变体，例如不同的提示词结构或后处理策略。详见[数据集配置](datasets.md)。

## 推理、评测与汇总

评测任务执行时，Partitioner 把“模型 × 数据集”拆成可并行任务 Task ，Runner 决定任务在本地、或其他的集群环境中如何执行，Task 执行实际推理或评测过程。例如基础教程的 API 示例中，使用 `OpenICLInferConcurrentTask` 进行高效并发推理，配合 `OpenICLEvalWatchTask` 完成结果实时评测。

推理步骤和评测步骤的结果会分别写入 `predictions/` 和 `results/`。Summarizer 再把各子集结果整理为最终的结果汇总文件。由于各步骤产出是分开保存的，因此当任务意外中断后，可以通过 `--reuse` 来从任务缺失阶段开始继续执行，或使用 `--mode eval`、`--mode viz` 对已有结果进行单步处理。详见[复用、恢复与分阶段执行](../execution/reuse_and_resume.md)，以及[理解输出与结果汇总](results_and_summarizer.md)。

## 工作目录与可复现性

未指定时，输出位于 `outputs/default/<时间戳>/`。建议每个实验显式指定稳定的 `--work-dir`，并保留其中的 `configs/` 配置快照，方便检查模型版本、数据版本、依赖环境、随机参数和 Judge 模型对结果产生的影响。

上述内容仅对opencompass工作流的各个步骤做粗略的描述。想要完整启动一次基于配置文件的评测，请参考[使用配置文件完成一次完整评测](config_based_evaluation.md)。
