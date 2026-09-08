# 任务恢复、复用与只重跑评测

OpenCompass 把预测、评分与汇总分开保存，因此可以复用已完成阶段。复用的前提是工作目录、时间戳以及模型/数据集简称和切分方式保持一致。

## 复用最近一次运行

```bash
opencompass my_eval.py --work-dir outputs/my_eval --reuse
```

不带值的 `--reuse` 等价于选择该工作目录下按名称排序的最新时间戳目录。生产环境更推荐显式指定时间戳：

```bash
opencompass my_eval.py \
    --work-dir outputs/my_eval \
    --reuse 20260903_120000
```

## 分阶段执行

```bash
# 只生成预测
opencompass my_eval.py --work-dir outputs/my_eval --mode infer

# 对已有预测重新评分
opencompass my_eval.py --work-dir outputs/my_eval \
    --reuse 20260903_120000 --mode eval

# 对已有结果重新汇总
opencompass my_eval.py --work-dir outputs/my_eval \
    --reuse 20260903_120000 --mode viz
```

`eval` 和 `viz` 模式必须通过 `--reuse` 指明已有实验，或使用结果站读取机制。

## 安全复用方案

- 只改变 Summarizer：通常可复用 results，运行 `viz`；
- 只改变答案抽取或 Evaluator：可复用 predictions，运行 `eval`；
- 改变 Prompt、few-shot、模型或生成参数：必须重新推理；
- 改变切分数量、简称或样本范围：应视为新实验，除非逐文件确认兼容。

不要仅因目标文件存在就假定内容有效。恢复后应检查失败日志、预测数量、结果数量和配置快照。
