# Prompt 模板

## 背景

在语言模型的评测中，我们常会将原始数据集以一定的规则构造成 prompt，以便模型能够按照要求回答问题。

通常，我们会在 prompt 开头放入指令，几个 in-context example（上下文样例），再在最后放入题目。例如：

```text
Solve the following questions.
1+1=?
2
3+9=?
12
5+6=?
```

大量的实验表明，即便测试的原始题目相同，对于 prompt 的不同构造方式会对模型的表现产生影响。可能影响的因素包括：

- Prompt 本身的构成方式，包括指令、in-context example、题目的写法；
- in-context example 的选择，包括了选择的数量和方式；
- 对 prompt 的使用方式。是让模型基于 prompt 进行补全，还是从候选的 prompt 中选择一个最好的作为答案？

OpenCompass 将 prompt 的构建策略定义在了数据集配置中的 `infer_cfg` 部分。一个典型的 `infer_cfg` 如下所示:

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

本文档中，我们将会主要介绍 `ice_template`、`prompt_template`、`inferencer` 的定义方法。对于 `retriever` 的介绍请参考其他章节。

我们首先介绍 prompt 的基本语法。

## 字符串式 prompt

字符串式的模板是比较经典的模板形式，考虑下面的模板：

```python
prompt_template=dict(
    type=PromptTemplate,
    template="{anything}\nQuestion: {question}\nAnswer: {answer}"
)
```

运行时，花括号`{}`内的字段会被替换成数据样本内的对应字段。如果数据样本中没有对应的字段，则会保持原样输出。

例如我们有一个数据 example 如下:

```python
example = {
    'question': '1+1=?',
    'answer': '2',  # 假设 answer 被写在了 reader_cfg.output_column 中
    'irrelavent_infos': 'blabla',
}
```

则填入模板后的结果为：

```text
{anything}
Question: 1+1=?
Answer:
```

可以看到，问题的实际答案 `answer` 并没有出现在生成的结果中。这是因为 OpenCompass 会遮盖被写在 `reader_cfg.output_column` 中的字段，避免答案泄露。关于 `reader_cfg` 的详细说明，请参考介绍数据集配置的相关文档。

## 对话式 prompt

在实际的测试中，简单的补全式测试并不能很好地测试出对话式的模型的性能，因此我们更希望 prompt 能以对话的格式输入到模型中。另外，不同的模型对对话的格式定义也不一样，因此我们也需要数据集侧产生的 prompt 更加通用，在测试时再结合具体模型生成符合需求的提示词。

因此，OpenCompass 在字符串式模板之上，增加了对对话式模板的支持。对话式模板更加灵活，它可以结合模型侧不同的 [meta_template](./meta_template.md) 生成不同对话形式的提示词，同时适用于基座和对话模型，但定义也相对复杂。

现在，让我们假设有一个数据样本如下：

```python
example = {
    'question': '1+1=?',
    'answer': '2',  # 假设 answer 被写在了 reader_cfg.output_column 中
    'irrelavent_infos': 'blabla',
}
```

接下来，我们来展示几个例子：

`````{tabs}

````{tab} 普通对话
```python
prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        round=[
            dict(role="HUMAN", prompt="Question: {question}"),
            dict(role="BOT", prompt="Answer: {answer}"),
        ]
    )
)
```

OpenCompass 把数据填入模板后得到的中间结果为：

```python
PromptList([
    dict(role='HUMAN', prompt='Question: 1+1=?'),
    dict(role='BOT', prompt='Answer: '),
])
```

````

````{tab} 多轮对话
```python
prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        round=[
            dict(role="HUMAN", prompt="Question: 2+2=?"),
            dict(role="BOT", prompt="Answer: 4"),
            dict(role="HUMAN", prompt="Question: 3+3=?"),
            dict(role="BOT", prompt="Answer: 6"),
            dict(role="HUMAN", prompt="Question: {question}"),
            dict(role="BOT", prompt="Answer: {answer}"),
        ]
    )
)
```

OpenCompass 把数据填入模板后得到的中间结果为：

```python
PromptList([
    dict(role='HUMAN', prompt='Question: 2+2=?'),
    dict(role='BOT', prompt='Answer: 4'),
    dict(role='HUMAN', prompt='Question: 3+3=?'),
    dict(role='BOT', prompt='Answer: 6'),
    dict(role='HUMAN', prompt='Question: 1+1=?'),
    dict(role='BOT', prompt='Answer: '),
])
```
````


````{tab} 带 SYSTEM 的对话

```python
prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        begin=[
            dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following questions.'),
        ],
        round=[
            dict(role="HUMAN", prompt="Question: {question}"),
            dict(role="BOT", prompt="Answer: {answer}"),
        ]
    )
)
```

OpenCompass 把数据填入模板后得到的中间结果为：

```python
PromptList([
    dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following questions.'),
    dict(role='HUMAN', prompt='Question: 1+1=?'),
    dict(role='BOT', prompt='Answer: '),
])
```

在具体的 meta template 中处理时，如果定义中存在 SYSTEM 角色，则会调用 SYSTEM 的模板进行处理。否则，会调用 fallback_role 角色的模板进行处理，也就是这个例子中的 HUMAN 角色。

````

`````

可以见到，在对话式的模板中，prompt 是以不同角色 `role` 的对话为形式进行组织的。在当前 OpenCompass 的预定义数据集配置中，一个 prompt 中常有的角色有：

- `HUMAN`：人类，通常为提问的一方
- `BOT`：语言模型，通常为回答的一方
- `SYSTEM`：系统，通常用在提示词的开头，负责下达指令。

另外与字符串式的模板不同，经过对话式模板所生成的 prompt 从固定的字符串变成了一个中间结构 PromptList。这个结构会进一步与模型侧的 [meta template](./meta_template.md) 相结合，拼装完成得到最终的提示词。如果不指定 meta template，PromptList 中各项的 prompt 则会直接按行拼接成字符串。

```{note}
上面例子中 PromptList 中的内容并非模型最终的输入，而取决于 meta template 的处理。一个容易产生误解的地方是，在生成式的评测中，最后一个 `BOT` 角色的 prompt `Answer: ` **不会**实际输入到模型。这是由于 API 模型通常并无法自定义模型回复的开头，因此这一设定保持了语言模型与 API 模型在评测上行为的一致。更多信息可以参考 [meta template](./meta_template.md) 的文档。
```

<details>
<summary>点击查看完整参数介绍</summary>

- `begin`，`end` ：(list，可选) prompt 的开头和结尾，通常是一些系统级别的指令。里面的每一项**允许是一个字典或字符串**。

- `round`：(list) 对话的模板格式。列表的每一项**只允许是一个字典**。

每一个字典的参数如下：

- `role`（str）: 参与对话的角色名，用于与 `meta_template` 中的名称进行关联，不会影响实际生成的 prompt。

- `fallback_role` (str) : 缺省角色名，假设 `meta_template` 中找不到 `role`，则会尝试使用 `fallback_role` 进行关联。默认为 `None`

- `prompt` (str) : 角色的对话内容。

</details>

## Prompt 模板 与 `inferencer`

在明白了 prompt 模板的基础定义方式后，我们还要根据 `inferencer` 的类型组织 prompt 模板。

OpenCompass 中主要支持了两种 Infernecer：`GenInferencer` 和 `PPLInferencer`，它们对应着两种不同的推理方式。

`GenInferencer` 对应生成式的推理。在推理时，模型被要求以输入的提示词为基准，继续往下续写。此时，`template` 则单一地表示这一句话对应的模板，例如:

`````{tabs}

````{group-tab} 字符串式模板
```python
prompt_template=dict(
    type=PromptTemplate,
    template='Solve the following questions.\n{question}\n{answer}'
)
```
````

````{group-tab} 对话式模板
```python
prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        begin=[
            dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following questions.'),
        ],
        round=[
            dict(role="HUMAN", prompt="{question}"),
            dict(role="BOT", prompt="{answer}"),
        ]
    )
)
```
````

`````

则模型的推理结果将会是往下续写的字符串。

而 `PPLInferencer` 对应判别式推理。在推理时，模型被要求计算多个输入字符串各自的混淆度 (PerPLexity / ppl)，并将其中 ppl 最小的项作为模型的推理结果。此时 `template` 是一个 `dict`，表示每一句话所对应的模板，例如:

`````{tabs}

````{group-tab} 字符串式模板
```python
prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        "A": "Question: Which is true?\nA. {A}\nB. {B}\nC. {C}\nAnswer: A",
        "B": "Question: Which is true?\nA. {A}\nB. {B}\nC. {C}\nAnswer: B",
        "C": "Question: Which is true?\nA. {A}\nB. {B}\nC. {C}\nAnswer: C",
        "UNK": "Question: Which is true?\nA. {A}\nB. {B}\nC. {C}\nAnswer: None of them is true.",
    )
)
```
````

````{group-tab} 对话式模板
```python
prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        "A": dict(
            round=[
                dict(role="HUMAN", prompt="Question: Which is true?\nA. {A}\nB. {B}\nC. {C}"),
                dict(role="BOT", prompt="Answer: A"),
            ]
        ),
        "B": dict(
            round=[
                dict(role="HUMAN", prompt="Question: Which is true?\nA. {A}\nB. {B}\nC. {C}"),
                dict(role="BOT", prompt="Answer: B"),
            ]
        ),
        "C": dict(
            round=[
                dict(role="HUMAN", prompt="Question: Which is true?\nA. {A}\nB. {B}\nC. {C}"),
                dict(role="BOT", prompt="Answer: C"),
            ]
        ),
        "UNK": dict(
            round=[
                dict(role="HUMAN", prompt="Question: Which is true?\nA. {A}\nB. {B}\nC. {C}"),
                dict(role="BOT", prompt="Answer: None of them is true."),
            ]
        ),
    )
)
```
````

`````

此时模型的推理结果将会是 `template` 的四个 key 之一 ("A" / "B" / "C" / "UNK")

## `ice_template` 与 `prompt_template`

在 OpenCompass 中，对于 0-shot 的评测，我们通常只需要定义 `prompt_template` 字段，即可完成 prompt 的构造。但对于 few shot 的评测，我们还需要定义 `ice_template` 字段，管理上下文学习中样例所对应的 prompt 模板。

`ice_template` 和 `prompt_template` 两者遵循的语法和规则一致，完整 prompt 的构造流程可以使用如下的伪代码进行表示：

```python
def build_prompt():
    ice = ice_template.format(*ice_example)
    prompt = prompt_template.replace(prompt_template.ice_token, ice).format(*prompt_example)
    return prompt
```

现在，让我们假设有两个训练数据 (ex1, ex2) 和一个测试数据 (ex3):

```python
ex1 = {
    'question': '2+2=?',
    'answer': '4',
    'irrelavent_infos': 'blabla',
}
ex2 = {
    'question': '3+3=?',
    'answer': '6',
    'irrelavent_infos': 'blabla',
}
ex3 = {
    'question': '1+1=?',
    'answer': '2',  # 假设 answer 被写在了 reader_cfg.output_column 中
    'irrelavent_infos': 'blabla',
}
```

接下来，我们看一下不同的 prompt 构造方法对应的实际效果：

`````{tabs}

````{group-tab} 字符串式模板

模板配置如下：

```python
infer_cfg=dict(
    ice_template=dict(
        type=PromptTemplate,
        template='{question}\n{answer}'
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template='Solve the following questions.\n</E>{question}\n{answer}'
        ice_token='</E>',
    )
)
```

会得到以下字符串：

```text
Solve the following questions.
2+2=?
4
3+3=?
6
1+1=?

```

````

````{group-tab} 对话式模板

模板配置如下：

```python
infer_cfg=dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt="{question}"),
                dict(role="BOT", prompt="{answer}"),
            ]
        )
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following questions.'),
                '</E>',
            ],
            round=[
                dict(role="HUMAN", prompt="{question}"),
                dict(role="BOT", prompt="{answer}"),
            ],
        ),
        ice_token='</E>',
    )
)
```

OpenCompass 把数据填入模板后得到的中间结果为：

```python
PromptList([
    dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following questions.'),
    dict(role='HUMAN', prompt='2+2=?'),
    dict(role='BOT', prompt='4'),
    dict(role='HUMAN', prompt='3+3=?'),
    dict(role='BOT', prompt='6'),
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt=''),
])
```
````

`````

### 省略式使用方法

值得一提的是，为了简便配置文件，`prompt_template` 这一字段是可被省略的。当 `prompt_template` 字段被省略时，`ice_template` 会同时被作为 `prompt_template`，用于拼装得到完整的 prompt。以下两份 `infer_cfg` 是等价的：

<table class="docutils">
  <thead>
  <tr>
      <th>完整写法</th>
      <th>省略写法</th>
  <tbody>
  <tr>
  <td>

```python
infer_cfg=dict(
    ice_template=dict(
        type=PromptTemplate,
        template="Q: {question}\nA: {answer}",
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template="</E>Q: {question}\nA: {answer}",
        ice_token="</E>",
    ),
    # ...
)
```

</td>
  <td>

```python
infer_cfg=dict(
    ice_template=dict(
        type=PromptTemplate,
        template="</E>Q: {question}\nA: {answer}",
        ice_token="</E>",
    ),
    # ...
)
```

</td>
  </tr>
  </thead>
  </table>

更一般地，即便在 0-shot learning 的情况下（即 `retriever` 为 `ZeroRetriver`）时，这一机制依然生效。因此以下配置也是合法的：

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

## 使用建议

建议使用 [Prompt Viewer](../tools.md) 工具对完成拼装后的 prompt 进行可视化，确认模板是否正确，结果是否符合预期。
