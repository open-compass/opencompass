# 提示词预览与调试

修改 Prompt 后不要直接跑完整基准。先预览模型最终接收的输入，再用少量样本验证生成与答案抽取。

## Prompt Viewer

```bash
python tools/prompt_viewer.py my_eval.py -n -c 3
```

- `-n`：非交互模式，选择第一个模型和数据集；
- `-c`：输出的样本数量；
- `-a`：检查所有模型—数据集组合；
- `-p PATTERN`：只选择简称匹配的 Dataset。

传入完整实验配置时，工具会构建 tokenizer 并展示模型模板处理后的输入，同时报告 token 数；只传数据集配置时，通常只能检查数据集侧 Prompt。

## 只导出消息

若评测后端或模板不适合 Prompt Viewer，可以使用 `--dump-only-message-path` 导出构造后的输入。该参数目前仅支持使用 `GenInferencer` 的数据集配置，建议同时指定 `--mode infer`，避免继续进入评测阶段：

```bash
opencompass my_eval.py \
    --mode infer \
    --dump-only-message-path /opencompass-messages \
    --debug
```

导出目录按照模型和数据集的 `abbr` 组织：

```text
/opencompass-messages/
└── gpt-6-astra-response/
    └── demo_gsm8k.jsonl
```

JSONL 文件每行对应一个样本：

```json
{"message": [{"role": "system", "content": "Answer concisely."}, {"role": "user", "content": "What is 1 + 1?"}], "gold": "2"}
```

`message` 是数据集模板完成字段替换和 few-shot 拼接后，再经过模型 `meta_template` 处理的结果；`gold` 是未经评测后处理的原始参考答案。导出发生在 `model.generate()` 之前，因此不会应用其内部的 tokenizer chat template、分词或 API 请求转换，也不会执行模型生成。任务仍会初始化模型对象，且不会产生可用于评分或复用的正式 `predictions/` 文件。
