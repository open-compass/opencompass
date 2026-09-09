# 提示词与输入构造总览

提示词配置决定怎样把一条数据样本转换成模型实际接收的输入。即使模型、原始数据和 Evaluator 相同，Prompt、few-shot 样例或模型协议不同也可能显著改变结果。

## 为什么 Prompt 构造影响结果

评测中通常会把原始数据按一定规则构造成 prompt：开头放指令，中间放几个 in-context example（上下文样例），最后放题目，例如：

```text
Solve the following questions.
1+1=?
2
3+9=?
12
5+6=?
```

大量实验表明，即便测试的原始题目相同，不同的 prompt 构造方式也会影响模型表现。可能的因素包括：

- Prompt 本身的构成方式，包括指令、in-context example、题目的写法；
- in-context example 的选择，包括数量和选择方式；
- 对 prompt 的使用方式：是让模型基于 prompt 进行补全，还是从候选 prompt 中选择困惑度最低的一项作为答案。

因此，OpenCompass 把 prompt 的构建策略定义在数据集配置的 `infer_cfg` 部分：

```python
infer_cfg=dict(
    ice_template=dict(  # 用于构造 In Context Example (ice) 的模板
        type=PromptTemplate,
        template='{question}\n{answer}'
    ),
    prompt_template=dict(  # 用于构造主干 prompt 的模板
        type=PromptTemplate,
        template='Solve the following questions.\n</E>{question}\n{answer}',
        ice_token="</E>"
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1]),  # 定义 in context example 的获取方式
    inferencer=dict(type=GenInferencer),  # 使用何种方式推理得到 prediction
)
```

## 两层结构

OpenCompass 将输入构造分为两层：

```text
数据样本
  ↓ 数据集侧 PromptTemplate
统一的文本或 role/content 消息
  ↓ 模型侧 MetaTemplate / tokenizer chat template
模型实际接收的输入
```

### 数据集侧模板

新建对话评测配置时，优先使用 [RawPromptTemplate](raw_prompt_template.md)。它直接用 `role` / `content` 消息列表表达输入，便于与 OpenAI 兼容接口和 chat template 对照。

[PromptTemplate](prompt_template.md) 是传统模板，仍适合已有配置、PPL 多候选模板及复杂的 in-context learning 样例拼接。它支持字符串式与对话式两种写法，后者经由中间结构 PromptList 与模型侧协议结合。

### 模型侧协议

经过 SFT 的对话模型在训练时就约定了对话格式——系统层级的指令、表示角色的标记和特殊 token，例如：

```text
Meta instruction: You are now a helpful and harmless AI assistant.
HUMAN: Hi!<eoh>
Bot: Hello! How may I assist you?<eob>
```

评测时需要按照约定的格式输入问题，模型才能发挥出应有的性能。API 模型存在类似情况：多数对话接口允许传入历史对话，部分还支持 SYSTEM 层级指令，评测时应尽量贴合 API 模型本身的多轮对话结构，而不是把所有内容塞进一段指令。

[MetaTemplate](meta_template.md) 将统一角色映射为具体模型的对话格式。部分模型改由 tokenizer 内置 chat template 完成同一工作。模型协议不应承载数据集题目内容，数据集模板也不应硬编码某个模型的特殊 token。

## 多轮和推理策略

[多轮输入](raw_prompt_template.md#多轮输入与推理)要求保留会话状态，后续输入依赖前一轮回复。[思维链与推理策略](chain_of_thought.md)会改变生成长度、答案抽取和采样方式，报告结果时必须同时记录。

## 调试原则

修改模板后，先使用 `tools/prompt_viewer.py` 检查若干条最终输入，再用小范围数据完整运行。至少核对字段替换、few-shot 顺序、角色映射、system prompt、特殊 token、截断位置和答案输出要求。
