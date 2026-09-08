# RawPromptTemplate：推荐的消息模板

`RawPromptTemplate` 直接使用常见的 `role` / `content` 消息列表描述模型输入。对于新建的对话模型数据集配置，优先采用这种形式：它更接近 OpenAI 兼容接口和多数 Chat Template 的输入结构，也更容易预览最终消息。

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

| 元素形式 | 作用 |
| --- | --- |
| `dict(role=..., content=...)` | 常规消息，`content` 中的 `{field}` 会被样本字段替换 |
| `dict(expand_column='xxx')` | 从样本的 `xxx` 字段读取一段消息列表，原位展开插入 |
| `'</E>'`（字符串） | 少样本示例（ICE）的插入位置，见下文 |

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

继续使用 [PromptTemplate](prompt_template.md) 的场景：

- 维护已有稳定配置；
- 需要比单一 `</E>` 插入点更复杂的少样本编排；
- 使用 PPL 等需要多候选模板的传统推理方式。

## 数据集模板与模型模板的边界

数据集侧的 RawPromptTemplate 表达“这道题怎样询问”，模型侧的 [MetaTemplate](meta_template.md) 或 tokenizer chat template 表达“这个模型怎样编码角色”。不要在两侧都写模型专用特殊 token，也不要无意中追加两份 system prompt。

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

## 从传统模板迁移

迁移时按以下顺序核对：

1. 把 `begin` 中的系统指令转换成 system 消息；
2. 把每一轮 HUMAN/BOT 内容转换成 user/assistant 消息；
3. 保留字段占位符，确认花括号转义正确；
4. 预览若干条最终输入；
5. 用小范围数据比较迁移前后的预测，确认少样本顺序、停止词和答案抽取没有变化。

模板格式更直观不代表评测语义自动等价。涉及 few-shot、重复采样或特殊模型协议时，迁移必须做逐样本对照。
