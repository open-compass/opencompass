# 重复输出与响应长度分析

推理时增加 `--dump-res-length` 可以把回复长度信息写入预测。汇总阶段增加 `--analysis-repeat` 可以分析异常重复模式：

```bash
opencompass my_eval.py --dump-res-length --analysis-repeat
```

分析已有结果可使用：

```bash
python tools/analyze_repeat.py outputs/my_eval/<时间戳> \
    --model model-abbr \
    --tokenizer gpt-4o
```

可用 `--think-tag` 分开分析推理内容和最终回复，`--out` 指定输出文件。若提供 Hugging Face tokenizer，工具可能需要联网加载；离线环境应传入本地 tokenizer 路径。

响应长度用于发现截断、空回复和成本异常，不等同于输入 token 统计。长上下文输入检查参阅[长上下文评测](../evaluation/long_context.md)。
