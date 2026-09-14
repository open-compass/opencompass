# 提示词模板：以 RawPromptTemplate 为默认

提示词配置决定怎样把一条数据样本转换成模型实际接收的输入。即使模型、原始数据和 Evaluator 相同，Prompt、few-shot 样例或模型协议不同也可能显著改变结果。

评测通常会在题目之前加入指令和 in-context example，再选择生成式补全或 PPL 等推理方式。因此 OpenCompass 把输入构造放在数据集配置的 `infer_cfg` 中，并将它分为两层：

```text
数据样本
  ↓ 数据集侧模板（RawPromptTemplate，默认 / PromptTemplate，传统）
统一的文本或 role/content 消息
  ↓ 模型侧对话模板协议（MetaTemplate）/ tokenizer chat template
模型实际接收的输入
```

新建对话评测配置时优先使用 `RawPromptTemplate`。它直接表达 `role/content` 消息，便于与 API 和 tokenizer chat template 对照。`PromptTemplate` 主要用于维护旧配置、PPL 多候选模板和复杂 ICE 编排；模型专属角色标记与特殊 token 则由[模型侧对话模板协议](meta_template.md)处理。

## RawPromptTemplate（默认）

## 基本用法

```python
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(role='system', content='You are a helpful assistant.'),
            dict(
                role='user',
                content=(
                    '{problem}\n'
                    'Put the final answer in \\boxed{{}}.'
                ),
            ),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

`{problem}` 来自数据集样本字段。模板渲染后得到消息列表，再交给模型后端。配置时应确认占位符都存在于 `reader_cfg.input_columns` 或检索得到的样本中。

## messages 的三类元素

`messages` 列表中允许出现三类元素：

| 元素形式                      | 作用                                                |
| ----------------------------- | --------------------------------------------------- |
| `dict(role=..., content=...)` | 常规消息，`content` 中的 `{field}` 会被样本字段替换 |
| `dict(expand_column='xxx')`   | 从样本的 `xxx` 字段读取一段消息列表，原位展开插入   |
| `'</E>'`（字符串）            | 少样本示例（ICE）的插入位置，见下文                 |

### expand_column：展开数据集中的消息列

当一段输入不是一两条消息，而是整段保存在数据集字段里的会话时，用 `expand_column` 把它整体插入。例如样本的 `dialogue` 字段为：

```python
# 数据集中的一行
{
    'dialogue': [
        {'role': 'user', 'content': '写一段产品介绍，不超过 100 词。'},
        {'role': 'assistant', 'content': ''},
        {'role': 'user', 'content': '把上一段改成更正式的语气。'},
        {'role': 'assistant', 'content': ''},
    ]
}
```

配置写作：

```python
reader_cfg = dict(input_columns=['dialogue'], output_column='reference')

infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[{'expand_column': 'dialogue'}],
        format_variables=False,
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

`expand_column` 元素没有 `role`/`content`，渲染时读取该字段并把其中每条消息按顺序插入。`format_variables=False` 表示这些消息是现成数据，不再做占位符替换——适合已经渲染好的对话和 ChatML 数据。

### Few-shot 插入示例（ICE）

主模板中的字符串 `'</E>'`（默认 `ice_token`）标记少样本示例的插入位置；每个示例的样式由单独的 `ice_template` 描述。以下配置摘自 `opencompass/configs/datasets/SciReasoner/mol_biotext_rawprompt_gen.py`：

```python
infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'system',
             'content': 'There is a single choice question about chemistry. '
                        'Answer the question directly.'},
            '</E>',
            {'role': 'user', 'content': 'Query: {input}'},
        ],
    ),
    ice_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': 'Query: {input}'},
            {'role': 'assistant', 'content': '{output}'},
        ],
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0]),
    inferencer=dict(type=GenInferencer),
)
```

渲染逻辑：

- 检索器选出 k 个少样本（`FixKRetriever` 固定取指定序号，也可换成 `TopkRetriever`、`RandomRetriever` 等）；
- `ice_template` 把每个样本渲染成一组消息（这里是 user 提问 + assistant 答案两条）；
- 所有示例消息按顺序替换主模板中 `'</E>'` 的位置，最终输入仍是单个 messages 列表：system + k 组示例 + 当前问题。

## 多轮输入与推理

多轮评测要求模型在同一会话中连续回答，并保留此前的用户消息和模型回复。其后续轮次的输入依赖前一轮生成结果，任务不能任意拆成彼此独立的样本。

数据侧通常这样组织：用户轮来自数据，助手轮 `content` 留空，作为等待生成的槽位；推理器逐轮推进，把生成结果写回助手轮并累计进上下文。以下是多轮指令遵循数据集 MultiIF 的完整 `infer_cfg`：

```python
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

multiif_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[{'expand_column': 'dialogue'}],
        format_variables=False,
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
        multiround=True,
    ),
)
```

`GenInferencer` 的 `multiround=True` 开启逐轮推理：遇到空的 assistant 轮就生成一次，生成内容会保留在上下文中供后续轮次使用。评测配置需要指明读取哪个角色的输出，MultiIF 使用 `eval_cfg=dict(evaluator=..., pred_role='BOT')`。若模型后端会自动插入 system 消息或 generation prompt，还要确认它没有与数据集模板重复

## 与 PromptTemplate 的区别

传统 `PromptTemplate` 使用 `begin`、`round`、`ice_token` 等结构表达角色和少样本拼接，再映射成模型输入。`RawPromptTemplate` 则让配置内容与最终消息一一对应。

优先使用 `RawPromptTemplate` 的场景：

- 新增面向对话模型或 OpenAI 兼容接口的数据集；
- 输入本身是 system/user/assistant 消息，或整段会话保存在数据集字段中；
- 希望减少角色映射并方便逐条检查消息。

继续使用 [PromptTemplate](#prompttemplate传统模板) 的场景：

- 维护已有稳定配置；
- 需要比单一 `</E>` 插入点更复杂的少样本编排；
- 使用 PPL 等需要多候选模板的传统推理方式。

## 数据集模板与模型模板的边界

数据集侧的 RawPromptTemplate 表达“这道题怎样询问”，模型侧的 [模型侧对话模板协议](meta_template.md) 或 tokenizer chat template 表达“这个模型怎样编码角色”。不要在两侧都写模型专用特殊 token，也不要无意中追加两份 system prompt。

部分 API 模型配置允许通过 `meta_template` 追加消息，例如：

```python
models = [
    dict(
        type=OpenAISDK,
        abbr='my-api-model',
        path='my-model',
        key='ENV',
        openai_api_base='https://example.com/v1',
        meta_template=[
            dict(role='system', content='Additional system instruction.'),
        ],
        max_seq_len=32768,
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=0),
    )
]
```

使用前请检查具体模型类如何读取 `key` 和 `meta_template`；不同 API 后端的字段并不完全相同。

## PromptTemplate（传统模板）

`PromptTemplate` 是 OpenCompass 的传统模板系统。新配置默认使用 RawPromptTemplate；维护已有配置、构造 PPL 多候选输入或进行复杂的 ICE 编排时，仍可使用下面的语法。

### 字符串式 prompt

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

### 对话式 prompt

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

由模型侧对话模板协议处理时，如果定义中存在 SYSTEM 角色，则会调用 SYSTEM 的模板进行处理。否则，会调用 fallback_role 角色的模板进行处理，也就是这个例子中的 HUMAN 角色。

````

`````

可以见到，在对话式的模板中，prompt 是以不同角色 `role` 的对话为形式进行组织的。在当前 OpenCompass 的预定义数据集配置中，一个 prompt 中常有的角色有：

- `HUMAN`：人类，通常为提问的一方
- `BOT`：语言模型，通常为回答的一方
- `SYSTEM`：系统，通常用在提示词的开头，负责下达指令。

另外与字符串式的模板不同，经过对话式模板所生成的 prompt 从固定的字符串变成了一个中间结构 PromptList。这个结构会进一步与[模型侧对话模板协议](meta_template.md)相结合，拼装完成得到最终的提示词。如果不指定模型侧协议，PromptList 中各项的 prompt 则会直接按行拼接成字符串。

```{note}
上面例子中 PromptList 中的内容并非模型最终的输入，而取决于模型侧对话模板协议的处理。一个容易产生误解的地方是，在生成式的评测中，最后一个 `BOT` 角色的 prompt `Answer: ` **不会**实际输入到模型。这是由于 API 模型通常并无法自定义模型回复的开头，因此这一设定保持了语言模型与 API 模型在评测上行为的一致。更多信息可以参考[模型侧对话模板协议](meta_template.md)文档。
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

### Prompt 模板 与 `inferencer`

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

### `ice_template` 与 `prompt_template`

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

#### 省略式使用方法

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

### 使用建议

建议使用 [Prompt Viewer](../tools/index.md) 工具对完成拼装后的 prompt 进行可视化，确认模板是否正确，结果是否符合预期。

## 从传统模板迁移

迁移时按以下顺序核对：

1. 把 `begin` 中的系统指令转换成 system 消息；
2. 把每一轮 HUMAN/BOT 内容转换成 user/assistant 消息；
3. 保留字段占位符，确认花括号转义正确；
4. 预览若干条最终输入；
5. 用小范围数据比较迁移前后的预测，确认少样本顺序、停止词和答案抽取没有变化。

模板格式更直观不代表评测语义自动等价。涉及 few-shot、重复采样或特殊模型协议时，迁移必须做逐样本对照。

## 调试

模板修改后应先预览最终输入，检查角色顺序、few-shot 插入位置、特殊 token 与截断情况。参阅[提示词调试与 Prompt Viewer](debugging.md)。
