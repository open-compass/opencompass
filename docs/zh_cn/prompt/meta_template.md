# 模型侧对话模板协议

`meta_template` 写在模型配置中，用于在数据集模板生成输入后，补充模型侧的额外指令或适配模型的对话格式。它同时兼容标准对话模版 `RawPromptTemplate` 和传统对话模版 `PromptTemplate`，但两种模板使用的配置结构不同：

| 数据集侧模板        | `meta_template` 结构 | 主要用途                               |
| ------------------- | -------------------- | -------------------------------------- |
| `RawPromptTemplate` | `role/content` 列表  | 插入消息，或在已有消息前追加指令       |
| `PromptTemplate`    | 包含 `round` 的字典  | 映射传统角色，或包装模型专属特殊 token |

两种结构不能混用。数据集特有的任务指令应优先写入数据集模板；只有希望同一个模型在所有评测中统一携带的内容，才适合写入模型侧 `meta_template`。

## 为标准模板追加内容

[标准提示词模板](raw_prompt_template.md#标准提示词模板)直接生成由 `system`、`user` 和 `assistant` 组成的消息列表。此时，模型配置中的 `meta_template` 也写成 `role/content` 列表：

```python
from opencompass.models import OpenAISDK

models = [
    dict(
        type=OpenAISDK,
        abbr='my-api-model',
        path='served-model-name',
        key='ENV',
        openai_api_base='https://example.com/v1',
        meta_template=[
            dict(
                role='system',
                content='Answer concisely and put the final answer last.\n',
            ),
        ],
        max_seq_len=32768,
        max_out_len=4096,
        batch_size=8,
    )
]
```

假设数据集侧生成以下消息：

```python
[
    {'role': 'system', 'content': 'Solve the following problem.'},
    {'role': 'user', 'content': 'What is 1 + 1?'},
]
```

模型侧模板处理后的输入为：

```python
[
    {
        'role': 'system',
        'content': (
            'Answer concisely and put the final answer last.\n'
            'Solve the following problem.'
        ),
    },
    {'role': 'user', 'content': 'What is 1 + 1?'},
]
```

处理规则如下：

- 当模型侧消息与数据集消息在对应位置具有相同 `role` 时，模型侧 `content` 会添加到数据集消息内容之前；
- 当后续数据集消息中不存在该 `role` 时，模型侧消息会作为一条新消息插入；
- 模型侧列表按照声明顺序处理，`role` 只能使用 `system`、`user` 或 `assistant`。

这种列表写法不会执行传统模板的角色映射，也不使用 `round`、`api_role`、`begin`、`end` 或 `generate`。它适用于能够直接处理标准消息列表的模型类型，例如 OpenAI 兼容 API 模型；其他模型类型是否支持，应以其模板解析实现为准。

## 为传统模板适配模型格式

传统 `PromptTemplate` 生成的是由 `SYSTEM`、`HUMAN`、`BOT` 等角色组成的中间结构。此时，`meta_template` 使用字典格式，负责把这些角色转换为 API 消息，或包装成本地语言模型要求的字符串。

### API 与 Chat Template 模型

对于 API 模型，以及通过 tokenizer chat template 构造输入的模型，只需配置角色映射：

```python
from opencompass.models import OpenAISDK

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='user'),
        dict(role='BOT', api_role='assistant', generate=True),
    ],
    reserved_roles=[
        dict(role='SYSTEM', api_role='system'),
    ],
)

models = [
    dict(
        type=OpenAISDK,
        abbr='my-api-model',
        path='served-model-name',
        key='ENV',
        openai_api_base='https://example.com/v1',
        meta_template=api_meta_template,
        max_seq_len=32768,
        max_out_len=4096,
        batch_size=8,
    )
]
```

- `role` 必须与传统数据集模板中的角色名称一致；
- `api_role` 指定转换后的消息角色；
- `generate=True` 表示该角色由模型生成。在生成式推理中，最后一轮对应的 `BOT` 内容不会作为输入发送；PPL 推理仍会保留完整候选内容；
- `reserved_roles` 声明不会固定出现在每一轮、但可能在 `begin` 或 `end` 中出现的角色，通常用于 `SYSTEM`。

如果传统数据集模板中的 `SYSTEM` 设置了 `fallback_role='HUMAN'`，而模型侧没有声明 `SYSTEM`，该消息会按 `HUMAN` 角色处理。

### 本地一站式评测的语言模型

对于直接接收字符串的本地语言模型，`begin` 和 `end` 用于为各角色添加模型要求的标记：

```python
lm_meta_template = dict(
    begin='Meta instruction: You are a helpful assistant.\n',
    round=[
        dict(
            role='HUMAN',
            begin='<HUMAN>: ',
            end='<eoh>\n',
        ),
        dict(
            role='BOT',
            begin='<BOT>: ',
            end='<eob>\n',
            generate=True,
        ),
    ],
    reserved_roles=[
        dict(
            role='SYSTEM',
            begin='<SYSTEM>: ',
            end='<eosys>\n',
        ),
    ],
)
```

这里的 `begin` 和 `end` 是模型协议的一部分，不是任务指令。OpenCompass 会按传统数据集模板中的角色顺序套用这些标记；在生成式推理中，输入会停在 `generate=True` 角色的 `begin` 之后，等待模型继续生成。

使用 tokenizer chat template 的模型通常只需要前一节的角色映射，不应再手写相同的特殊 token，否则可能造成模板重复。

### 面向传统模板场景的支持字段

`meta_template` 字典支持以下主要字段：

| 字段             | 作用                                                               |
| ---------------- | ------------------------------------------------------------------ |
| `round`          | 必填；定义一轮对话中的角色顺序及每个角色的转换规则                 |
| `reserved_roles` | 可选；声明可能出现在轮次之外的角色                                 |
| `begin`、`end`   | 可选；为完整输入添加全局开头或结尾，主要用于直接接收字符串的模型   |
| `eos_token_id`   | 可选；为部分本地模型指定生成终止 token，具体支持情况取决于模型类型 |

`round` 和 `reserved_roles` 中的角色项支持：

| 字段           | 作用                                                 |
| -------------- | ---------------------------------------------------- |
| `role`         | 与传统 `PromptTemplate` 中的角色名称匹配             |
| `api_role`     | 将角色映射为 API 或 chat template 使用的角色         |
| `begin`、`end` | 在直接文本输入中包裹该角色的内容                     |
| `prompt`       | 可选的默认内容；数据集模板提供同一角色内容时会覆盖它 |
| `generate`     | 标记由模型生成的角色；通常只为 `BOT` 设置 `True`     |

同一个角色只能在 `round` 和 `reserved_roles` 中声明一次。面向 API 或 chat template 的配置使用 `api_role`；面向直接文本输入的配置使用角色级 `begin` 和 `end`，不应把两套写法混在同一个角色定义中。

## 不配置 meta_template 时

- `RawPromptTemplate` 生成的标准消息会保持原样；
- 传统 `PromptTemplate` 生成的带角色结构会失去角色包装，各项 `prompt` 通常仅按顺序拼接为字符串。

因此，API 和 chat template 评测优先使用 `RawPromptTemplate`；只有兼容传统数据集配置时才需要角色映射。直接加载依赖专用对话格式的本地模型时，应确保模型配置提供了正确的 `meta_template`。
