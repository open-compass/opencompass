# 准备模型

要在 OpenCompass 中支持新模型的评测，有以下几种方式：

1. 基于 HuggingFace 的模型
2. 基于 API 的模型
3. 自定义模型

## 基于 HuggingFace 的模型

在 OpenCompass 中，我们支持直接从 Huggingface 的 `AutoModel.from_pretrained` 和
`AutoModelForCausalLM.from_pretrained` 接口构建评测模型。如果需要评测的模型符合 HuggingFace 模型通常的生成接口，
则不需要编写代码，直接在配置文件中指定相关配置即可。

如下，为一个示例的 HuggingFace 模型配置文件：

```python
# 使用 HuggingFace 评测 HuggingFace 中 AutoModel 支持的模型
# 使用 HuggingFaceCausalLM 评测 HuggingFace 中 AutoModelForCausalLM 支持的模型
from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        # 以下参数为 `HuggingFaceCausalLM` 的初始化参数
        path='huggyllama/llama-7b',
        tokenizer_path='huggyllama/llama-7b',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        # 以下参数为各类模型都有的参数，非 `HuggingFaceCausalLM` 的初始化参数
        abbr='llama-7b',            # 模型简称，用于结果展示
        max_out_len=100,            # 最长生成 token 数
        batch_size=16,              # 批次大小
        run_cfg=dict(num_gpus=1),   # 运行配置，用于指定资源需求
    )
]
```

对以上一些参数的说明：

- `padding_side='left'`：在左侧进行填充，因为不是所有模型都支持填充，在右侧进行填充可能会干扰模型的输出。
- `truncation_side='left'`：在左侧进行截断，评测输入的 prompt 通常包括上下文样本 prompt 和输入 prompt 两部分，如果截断右侧的输入 prompt，可能导致生成模型的输入和预期格式不符，因此如有必要，应对左侧进行截断。

# 基于 API 的模型

OpenCompass 目前支持以下基于 API 的模型推理：

- OpenAI（`opencompass.models.OpenAI`）
- Comming soon

以下，我们以 OpenAI 的配置文件为例，模型如何在配置文件中使用基于 API 的模型。

```python
from opencompass.models import OpenAI

models = [
    dict(
        type=OpenAI,                             # 使用 OpenAI 模型
        # 以下为 `OpenAI` 初始化参数
        path='gpt-4',                            # 指定模型类型
        key='YOUR_OPENAI_KEY',                   # OpenAI API Key
        max_seq_len=2048,                        # 最大输入长度
        # 以下参数为各类模型都有的参数，非 `OpenAI` 的初始化参数
        abbr='GPT-4',                            # 模型简称
        run_cfg=dict(num_gpus=0),                # 资源需求（不需要 GPU）
        max_out_len=512,                         # 最长生成长度
        batch_size=1,                            # 批次大小
    ),
]
```

# 自定义模型

如果以上方式无法支持你的模型评测需求，请参考 [支持新模型](../advanced_guides/new_model.md) 在 OpenCompass 中增添新的模型支持。
