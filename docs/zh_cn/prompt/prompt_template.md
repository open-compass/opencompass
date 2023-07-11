# Prompt 模板

## 背景

Prompt 模板表征了原始数据集是如何被转化成能够被传入进模型的 `str` (或传入 API 模型的 `dict`) 的过程，它一般会被放置于 dataset 的 `infer_cfg` 字段。一个典型的 `infer_cfg` 如下所示:

```python
datasets = [
    dict(
        infer_cfg=dict(
            ice_template=dict(  # 用于构造 In Context Example (ice) 的模板
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role="HUMAN", prompt="Q: {question}"),
                        dict(role="BOT", prompt="A: {answer}"),
                    ]
                )
            ),
            prompt_template=dict(  # 用于构造主干 prompt 的模板
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(role="HUMAN", prompt="Suppose you are a student in the colledge entrance examination, answer the following questions."),
                        dict(role="BOT", prompt="OK, I am ready."),
                        "</E>",
                    ],
                    round=[
                        dict(role="HUMAN", prompt="Q: {question}"),
                        dict(role="BOT", prompt="A: {answer}"),
                    ]
                ),
                ice_token="</E>"
            ),
            retriever=dict(type=FixKRetriever),  # 构造 in context example 的编号是如何获取的
            inferencer=dict(type=GenInferencer, fix_id_list=[0, 1, 2, 3, 4]),  # 使用何种方式推理得到 prediction
        )
    ),
]
```

本文档中，我们将会主要讨论 `ice_template` 和 `prompt_template` 的相关使用方法。

## `prompt_template` 与 `inferencer`

`prompt_template` 首先需要与 `inferencer` 的类型契合。

一方面，对于生成式的 `GenInferencer` (推理时模型被要求以输入的提示词为基准，继续往下续写)，其 `template` 则单一地表示这一句话对应的模板，例如:

```python
ice_template=dict(
    type=PromptTemplate,
    template=dict(
        round=[
            dict(role="HUMAN", prompt="Question: {question}\nAnswer: ")
        ]
    )
)
```

则模型的推理结果将会是往下续写的字符串。

另一方面，对于判别式的 `PPLInferencer` (推理时模型被要求计算多个输入字符串各自的混淆度 / PerPLexity / ppl, 将其中 ppl 最小的项作为模型的推理结果)，其 `template` 则为一个 `dict`，表示每一句话所对应的模板，例如:

```python
ice_template=dict(
    type=PromptTemplate,
    template=dict(
        "A": dict(round=[dict(role="HUMAN", prompt="Question: Which is true?\nA. {A}\nB. {B}\nC. {C}\nAnswer: A")]),
        "B": dict(round=[dict(role="HUMAN", prompt="Question: Which is true?\nA. {A}\nB. {B}\nC. {C}\nAnswer: B")]),
        "C": dict(round=[dict(role="HUMAN", prompt="Question: Which is true?\nA. {A}\nB. {B}\nC. {C}\nAnswer: C")]),
        "UNK": dict(round=[dict(role="HUMAN", prompt="Question: Which is true?\nA. {A}\nB. {B}\nC. {C}\nAnswer: None of them is true.")]),
    )
)
```

则模型的推理结果将会是 `template` 的四个 key 之一 ("A" / "B" / "C" / "UNK")

## `prompt_template` 与 `meta_template`

一句话对应的模板可以是一个字符串，也可以是一个符合 [meta_template](./meta_template.md) 规范的字典。

### `str` 形式

我们首先来看字符串的例子:

```python
ice_template=dict(
    type=PromptTemplate,
    template="Question: {question}\nAnswer: {answer}"
)
```

我们会使用 python 中 `.format` 的方法填入模板。例如我们有一个数据 example 如下:

```python
example = {
    'question': '1+1=?',
    'answer': '2',  # 假设 answer 被写在了 reader_cfg.output_column 中
    'irrelavent_infos': 'blabla',
}
```

则填入模板后的结果为：

```text
Question: 1+1=?\nAnswer:
```

注意，由于 `answer` 被写到了 `reader_cfg.output_column` 中，因此它是不会在主干 `template` 中被替换掉的 (但仍然会在 ice template 中被替换掉)，也因此无需考虑答案泄漏的问题。

另外，当输入模型的 `meta_template` 非空，且数据集的 `template` 采用 `str` 形式的话，则拼装完成的 prompt 会对模型 `meta_template` 中 `HUMAN` 角色的 `prompt` 进行替换。

### `dict` 形式

我们再看一个 符合 [meta_template](./meta_template.md) 规范的 `dict` 的例子:

```python
ice_template=dict(
    type=PromptTemplate,
    template=dict(
        round=[
            dict(role="HUMAN", prompt="Q: {question}"),
            dict(role="BOT", prompt="A: {answer}"),
        ]
    )
)
```

`dict` 形式中的各项，我们仍然会使用类似于 python 中 `.format` 的方法填入模板。例如我们有一个数据 example 如下:

```python
example = {
    'question': '1+1=?',
    'answer': '2',  # 假设 answer 被写在了 reader_cfg.output_column 中
    'irrelavent_infos': 'blabla',
}
```

则填入模板后的结果为：

```text
PromptList([
    dict(role='HUMAN', prompt='Q: 1+1=?'),
    dict(role='BOT', prompt='A: '),
])
```

该结果会进一步与 meta template 机制相结合，最终拼装完成得到最终送入模型的字符串。具体见 [meta_template](./meta_template.md)

## `ice_template` 与 `prompt_template`

`ice_template` 和 `prompt_template` 一起组成拼装数据集所需的模板。`ice_template` 中的 `ice` 意为上下文学习样例 (In Context Example)。`ice_template` 负责构成上下文学习中的样例所对应的 prompt (few shot)。而 `prompt_template` 负责构成主干部分的 prompt。

完整 prompt 的构造流程可以使用如下的伪代码进行表示：

```python
def build_prompt():
    ice = ice_template.format(*ice_example)
    prompt = prompt_template.replace(ice_token, ice).format(*prompt_example)
    return prompt
```

一个样例如下：

```python
datasets = [
    dict(
        infer_cfg=dict(
            ice_template=dict(
                type=PromptTemplate,
                template="Q: {question}\nA: {answer}",
            ),
            prompt_template=dict(
                type=PromptTemplate,
                template="Suppose you are a math expert, answer the following question:\n</E>Q: {question}\nA: {answer}",
                ice_token="</E>"
            ),
            retriever=dict(type=FixKRetriever),
            inferencer=dict(type=GenInferencer, fix_id_list=[0, 1]),
        )
    ),
]
```

假设此时有

```python
ice_examples=[
    {"question": "1+1=?", "answer": "2"},
    {"question": "1-1=?", "answer": "0"},
]
example={"question": "54321**2+12345*67890=?", "answer": "3788873091"}
```

则最终输出的 prompt 是

```text
Suppose you are a math expert, answer the following question:
Q: 1+1=?
A: 2
Q: 1-1=?
A: 0
Q: 54321**2+12345*67890=?
A:
```

值得一提的是，为了简便配置文件，`prompt_template` 这一字段是可被省略的。当 `prompt_template` 字段被省略时，`ice_template` 会同时被作为 `prompt_template`，用于拼装得到完整的 prompt。以下两份配置文件是等价的：

```python
datasets = [
    dict(
        infer_cfg=dict(
            ice_template=dict(
                type=PromptTemplate,
                template="</E>Q: {question}\nA: {answer}",
                ice_token="</E>",
            ),
            retriever=dict(type=FixKRetriever),
            inferencer=dict(type=GenInferencer, fix_id_list=[0, 1, 2, 3, 4]),
        )
    ),
]
```

```python
datasets = [
    dict(
        infer_cfg=dict(
            ice_template=dict(
                type=PromptTemplate,
                template="Q: {question}\nA: {answer}",
            ),
            meta_template=dict(
                type=PromptTemplate,
                template="</E>Q: {question}\nA: {answer}",
                ice_token="</E>",
            )
            retriever=dict(type=FixKRetriever),
            inferencer=dict(type=GenInferencer, fix_id_list=[0, 1, 2, 3, 4]),
        )
    ),
]
```

## 使用建议

建议使用 [Prompt Viewer](../tools.md) 工具对完成拼装后的 prompt 进行可视化，确认模板是否正确，结果是否符合预期。
