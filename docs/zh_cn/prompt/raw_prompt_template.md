# 提示词模板

提示词配置决定怎样把一条数据样本转换成模型实际接收的输入。即使模型、原始数据和 Evaluator 相同，Prompt、few-shot 样例或模型协议不同也可能显著改变结果。

评测通常会在题目之前加入指令和 in-context example，再选择生成式补全或 PPL 等推理方式。因此 OpenCompass 把输入构造放在数据集配置的 `infer_cfg` 中，并将它分为两层：

```text
数据样本
  ↓ 数据集侧模板（RawPromptTemplate，标准 / PromptTemplate，传统）
统一的文本或 role/content 消息
  ↓ 模型侧对话模板协议（MetaTemplate）/ tokenizer chat template
模型实际接收的输入
```

标准提示词模板 `RawPromptTemplate` 直接表达 `role/content` 消息，适合新建对话或 API 评测配置；传统模板 `PromptTemplate` 主要用于兼容一站式部署与评测、 以及采用 PPL 等推理方式的基座模型评测。此外，还支持从模型配置侧向输入模板追加内容，请参阅[模型侧对话模板协议](meta_template.md)。

## 标准提示词模板

### 基本用法

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

### messages 的三类元素

`messages` 列表中允许出现三类元素：

| 元素形式                      | 作用                                                |
| ----------------------------- | --------------------------------------------------- |
| `dict(role=..., content=...)` | 常规消息，`content` 中的 `{field}` 会被样本字段替换 |
| `dict(expand_column='xxx')`   | 从样本的 `xxx` 字段读取一段消息列表，原位展开插入   |
| `'</E>'`（字符串）            | 少样本示例（ICE）的插入位置，见下文                 |

#### expand_column：展开数据集中的消息列

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

#### Few-shot 插入示例（ICE）

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

### 多轮输入与推理

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

`GenInferencer` 的 `multiround=True` 开启逐轮推理：遇到空的 assistant 轮就生成一次，生成内容会保留在上下文中供后续轮次使用。

## 传统提示词模板

传统的 `PromptTemplate` 主要用于兼容既有的一站式部署与评测配置，以及采用 PPL 推理方式的基座模型评测。新建对话或 API 评测配置时，优先使用前文介绍的 `RawPromptTemplate`。

`template` 可以是字符串，也可以通过 `begin`、`round` 和 `end` 组织带角色的传统对话模板；每条消息使用 `role` 和 `prompt` 描述，`fallback_role` 用于在模型侧没有对应 `role` 时指定替代 `role` 。带角色的模板可结合[模型侧对话模板协议](meta_template.md)转换为模型输入；未指定模型侧模板时，各项 `prompt` 会按顺序拼接。

常规生成任务可以按下面的方式配置：

```python
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt='You are a helpful assistant.',
                ),
            ],
            round=[
                dict(role='HUMAN', prompt='Question: {question}'),
                dict(role='BOT', prompt='Answer: {answer}'),
            ],
            end=[],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

`begin` 放置系统指令，`round` 描述用户输入和模型回复位置，`end` 表示对话结束后的附加内容。生成任务通常在最后一个 `BOT` 角色处开始生成，因此 `end` 一般留空或省略；当 `{answer}` 对应 `reader_cfg.output_column` 时，该字段会在推理时置空，避免参考答案泄露。

与生成式推理不同，`PPLInferencer` 要求 `template` 以字典形式为每个候选答案提供一份完整输入，并将 PPL 最低的候选项作为预测结果。示例如下：

```python
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A': (
                'Question: {question}\n'
                'A. {A}\nB. {B}\nC. {C}\n'
                'Answer: A'
            ),
            'B': (
                'Question: {question}\n'
                'A. {A}\nB. {B}\nC. {C}\n'
                'Answer: B'
            ),
            'C': (
                'Question: {question}\n'
                'A. {A}\nB. {B}\nC. {C}\n'
                'Answer: C'
            ),
        },
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)
```

构造 few-shot 输入时，使用与 `prompt_template` 语法相同的 `ice_template` 描述检索到的示例，并通过 `ice_token` 指定插入位置：

```python
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever

infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Question: {question}'),
                dict(role='BOT', prompt='Answer: {answer}'),
            ],
        ),
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt='Answer the following questions.',
                ),
                '</E>',
            ],
            round=[
                dict(role='HUMAN', prompt='Question: {question}'),
                dict(role='BOT', prompt='Answer: {answer}'),
            ],
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1]),
    inferencer=dict(type=GenInferencer),
)
```

`FixKRetriever` 从索引集固定选取序号为 0 和 1 的样本，`ice_template` 将它们渲染为两组问答，再替换 `prompt_template` 中的 `</E>`。当前待评测样本仍由 `prompt_template.round` 渲染，其参考答案会在推理时置空。
