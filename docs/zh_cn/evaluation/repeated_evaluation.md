# 循环评测、重复采样与稳定性

100 条题目的数据集不意味着框架会自动重复评测。默认每个 Dataset 只运行一次；只有配置中的 `n`、CLI 的 `--dataset-num-runs` 或专门的重复采样配置才会产生多次预测。

## 在配置文件中设置重复次数

在数据集 dict 的顶层直接写入 `n`，声明每个样本的独立运行次数；计算 pass@k 类指标时通常同时设置 `k`。仓库内 humaneval-plus 的重复采样配置就是这种写法：

```python
humaneval_plus_datasets = [
    dict(
        abbr='humaneval_plus',
        type=HumanevalDataset,
        path='opencompass/humaneval',
        reader_cfg=humaneval_plus_reader_cfg,
        infer_cfg=humaneval_plus_infer_cfg,
        eval_cfg=humaneval_plus_eval_cfg,
        n=5,  # 每个样本独立运行 5 次（默认 1）
        k=3,  # 传给评测器的 pass@k 档位，仅评测器支持时有意义
    )
]
```

两个字段的语义：

- `n`：重复次数。评测任务会把 `n` 连同预测一起交给 `evaluator.evaluate(k, n, ...)`，由评测器解释；`BaseEvaluator` 的默认实现是把 n 份预测逐份评分后取平均（`n>1` 时指标名带 `(5 runs average)` 后缀），并把同一样本的多次结果聚成一组，用于计算 pass@k、多次一致率等跨次指标。
- `k`：透传给评测器的 pass@k 档位，单个整数或列表（如 `k=[1, 10, 100]`），具体含义由 Evaluator 决定。

设置 `n` 有两个前提：

1. **推理侧要真的产生 n 倍预测**。`n` 只在评测阶段生效，不会自动让模型多采样：模型支持多回复时，在 `generation_kwargs` 中设 `num_return_sequences=n`；模型不支持时，用数据集加载器的 `num_repeats` 把每个样本复制 n 次（humaneval、apps 等数据集支持）。两种方式的完整配置见[代码评测](code_eval.md)的 pass@k 一节。
2. **评测器要能解释 `n` / `k`**（如 `HumanEvalPlusEvaluator`、`MBPPPassKEvaluator`）。普通 accuracy 类评测器即使接受 `n`，输出也只是多次平均，未必是需要的口径。

## 用命令行批量覆盖

```bash
opencompass my_eval.py --dataset-num-runs 5
```

等价于把配置里每个数据集的 `n` 和 `k` 批量改写为 5，适合临时对比；需要固定口径的正式评测应把 `n` / `k` 直接写进配置文件。注意该开关要求每个数据集都已定义 `n` 字段，否则会报错；且不能假设所有指标都输出同一种 pass@k 口径。

## 什么时候需要重复

- 生成使用非零温度或其他随机采样；
- API 服务存在不可控波动；
- 代码 pass@k、G-Pass@k 等指标明确要求多个候选；
- 研究 Prompt 或 Judge 的稳定性。

贪心解码且后端确定时，重复运行通常只增加成本。报告时至少给出运行次数、随机参数、每次分数、均值以及离散程度。

## 重复输出分析

正式运行时可增加：

```bash
opencompass my_eval.py --analysis-repeat
```

也可以分析已有时间戳目录：

```bash
python tools/analyze_repeat.py outputs/my_eval/20260903_120000 \
    --model model-abbr \
    --tokenizer gpt-4o
```

该工具分析回复内部异常重复模式，不等同于“多次评测的一致性分析”。tokenizer 会影响重复片段统计，应记录其名称。
