# 长上下文评测

长上下文评测的核心不是配置文件名中的 `32k` 或 `128k`，而是模型最终实际接收的 token 数、截断行为和可用输出预算。

OpenCompass 已包含 NeedleBench、RULER、LongBench 等长上下文配置。新配置优先选择包含 `rawprompt` 的变体，并审阅对应 README 和 Summarizer。

## 长度预算

```text
最终上下文 = system + few-shot + 题目正文 + 历史消息 + generation prompt
可用输入上限 ≈ max_seq_len - max_out_len
```

不同 tokenizer 对同一文本的 token 数不同。必须使用被评测模型的 tokenizer 检查最终消息；字符数、文件大小和其他模型 tokenizer 的结果只能作为估计。

## 常见失败

- tokenizer 或后端静默截断输入；
- 推理服务的真实上下文上限低于模型配置；
- `max_out_len` 占用过多上下文预算；
- 图片、工具消息或 chat template 带来额外 token；
- 超长样本导致显存溢出或请求超时；
- 分片配置改变后仍错误复用旧预测。

先通过 [Prompt Viewer](../prompt/debugging.md) 检查 token 数和截断，再按长度区间小规模试跑。报告应包含 tokenizer、输入 token 分布、最大值、截断样本数、输出预算与实际后端限制。
