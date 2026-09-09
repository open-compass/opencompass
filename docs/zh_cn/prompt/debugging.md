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

若评测后端或模板不适合 Prompt Viewer，可让推理任务只导出消息：

```bash
opencompass my_eval.py \
    --dump-only-message-path /tmp/opencompass-messages \
    --debug
```

该模式用于检查构造结果，不应当作正式预测。导出目录应使用专门位置，避免与实验结果混淆。

## 检查清单

- 所有字段占位符均已替换；
- system/user/assistant 的顺序符合预期；
- 数据集模板和模型模板没有重复指令；
- few-shot 样例没有泄露测试答案；
- 最终输入未超过模型上下文，截断没有删除题目；
- 生成提示、停止词和答案格式一致；
- API 和本地模型接收到的语义相同。

RawPromptTemplate 参阅[推荐的消息模板](raw_prompt_template.md)，模型协议参阅[MetaTemplate](meta_template.md)。
