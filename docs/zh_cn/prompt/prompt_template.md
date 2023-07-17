# Prompt 模板

## 背景

Prompt 模板定义了将原始数据集转化成提示词的过程，它一般会被放置于 dataset 的 `infer_cfg` 字段。一个典型的 `infer_cfg` 如下所示:

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

OpenCompass 中主要支持了两种 Infernecer：`GenInferencer` 和 `PPLInferencer`，它们对应着两种不同的推理方式。

`GenInferencer` 对应生成式的推理。在推理时，模型被要求以输入的提示词为基准，继续往下续写。此时，`template` 则单一地表示这一句话对应的模板，例如:

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

`PPLInferencer` 对应判别式推理。在推理时，模型被要求计算多个输入字符串各自的混淆度 (PerPLexity / ppl)，并将其中 ppl 最小的项作为模型的推理结果。此时 `template` 是一个 `dict`，表示每一句话所对应的模板，例如:

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

### 字符串式模板

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

```{warning}
纯字符串模板只适用于模型 meta_template 为空的状况。
```

### 对话式模板

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

对话式模板中的各项，我们仍然会使用类似于 python 中 `.format` 的方法填入模板。例如我们有一个数据 example 如下:

```python
example = {
    'question': '1+1=?',
    'answer': '2',  # 假设 answer 被写在了 reader_cfg.output_column 中
    'irrelavent_infos': 'blabla',
}
```

则填入模板后的结果为：

```python
PromptList([
    dict(role='HUMAN', prompt='Q: 1+1=?'),
    dict(role='BOT', prompt='A: '),
])
```

该结果会进一步与 meta template 机制相结合，最终拼装完成得到最终送入模型的字符串。若有 `meta_template` 如下:

```python
meta_template=dict(
    begin='Meta instruction: You are now a helpful and harmless AI assistant.\n',
    round=[
        dict(role='HUMAN', begin='<HUMAN>: ', end=''),
        dict(role='BOT', begin='', end='<eob>\n', generate=True),
    ],
    end='end of conversation',
)
```

则填入模板后的结果 `PromptList` 会根据 `role` 依次找到 `meta_template` 里 `round` 的对应条目 (注意 `meta_template` 中 `round` 对应的 `role` 是不会重复的)，并将 `prompt` 带入进去，得到一个新的 `PromptList`

```python
PromptList([
    '<HUMAN>: Q: 1+1=?<eoh>\n',
    '<BOT>: A: ',
])
```

最终联合上 `meta_template` 的 `begin` 与 `end` 字段，得到以下字符串

```text
Meta instruction: You are now a helpful and harmless AI assistant.
<HUMAN>: Q: 1+1=?<eoh>
<BOT>: A:
```

有以下示意图：

![](https://user-images.githubusercontent.com/22607038/251195073-85808807-6359-44df-8a19-9f5d00c591ec.png)

对话式模板的完整参数介绍如下：

- `role`（str）: 参与对话的角色名，用于与 `meta_template` 中的名称进行关联，不会影响实际生成的 prompt。
- `fallback_role` (str) : 缺省角色名，假设 `meta_template` 中找不到 `role`，则会尝试使用 `fallback_role` 进行关联。默认为 `None`
- `prompt` (str) : 角色的对话内容。

## `ice_template` 与 `prompt_template`

`ice_template` 和 `prompt_template` 一起组成拼装数据集所需的模板。`ice_template` 中的 `ice` 意为上下文学习样例 (In Context Example)。`ice_template` 负责构成上下文学习中的样例所对应的 prompt (few shot)。而 `prompt_template` 负责构成主干部分的 prompt。

完整 prompt 的构造流程可以使用如下的伪代码进行表示：

```python
def build_prompt():
    ice = ice_template.format(*ice_example)
    prompt = prompt_template.replace(ice_token, ice).format(*prompt_example)
    return prompt
```

### 字符串式模板案例

一个字符串式模板的样例如下：

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

### 对话式模板案例

另外有对话式模板的样例如下：

```python
datasets = [
    dict(
        infer_cfg=dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role="HUMAN", prompt="Q: {question}"),
                        dict(role="BOT", prompt="A: {answer}"),
                    ]
                )
            ),
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(role="HUMAN", prompt="Suppose you are a math expert, answer the following question:"),
                        "</E>",
                    ],
                    round=[
                        dict(role="HUMAN", prompt="Q: {question}"),
                        dict(role="BOT", prompt="A: {answer}"),
                    ]
                ),
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

则输出的 `PromptList` 将会是

```python
PromptList([
    dict(role='SYSTEM', fallback_role='HUMAN', prompt='Suppose you are a math expert, answer the following question:'),
    PromptList([
        dict(role="HUMAN", prompt="Q: 1+1=?"),
        dict(role="BOT", prompt="A: 2"),
        dict(role="HUMAN", prompt="Q: 1-1=?"),
        dict(role="BOT", prompt="A: 0"),
    ]),  # 对应 </E>
    dict(role="HUMAN", prompt="Q: 54321**2+12345*67890=?"),
    dict(role='BOT', prompt='A: '),
])
```

若有 `meta_template` 如下:

```python
meta_template=dict(
    begin='Meta instruction: You are now a helpful and harmless AI assistant.\n',
    round=[
        dict(role='HUMAN', begin='<HUMAN>: ', end=''),
        dict(role='BOT', begin='', end='<eob>\n', generate=True),
    ],
    end='end of conversation',
)
```

则最终组合生成得到的字符串如下：

```text
Meta instruction: You are now a helpful and harmless AI assistant.
<HUMAN>: Suppose you are a math expert, answer the following question:<eoh>
<HUMAN>: Q: 1+1=?<eoh>
<BOT>: A: 2<eob>
<HUMAN>: Q: 1-1=?<eoh>
<BOT>: A: 0<eob>
<HUMAN>: Q: 54321**2+12345*67890=?<eoh>
<BOT>: A:
```

有以下几点需要特别说明：

* prompt_template 支持 `begin` / `round` / `end` 成员变量，分别表示对话开始时 / 过程中 / 结束时所用的 prompt。三者的类型均为 `list`。
* `begin` 和 `end` 的 `list` 中支持 `dict` 和 `str` 混合表示。

### 省略式使用方法

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

更一般地，当不存在 `ice` 或使用 `ZeroRetriver` 时，`prompt_template` 被省略的机制依然是会运作的，因此会有如下配置：

```python
datasets = [
    dict(
        infer_cfg=dict(
            ice_template=dict(
                type=PromptTemplate,
                template="Q: {question}\nA: {answer}",
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        )
    ),
]
```

这也是前述文档中我们使用 `ice_template` 而不是 `prompt_template` 作为案例的原因。

## 使用建议

建议使用 [Prompt Viewer](../tools.md) 工具对完成拼装后的 prompt 进行可视化，确认模板是否正确，结果是否符合预期。
