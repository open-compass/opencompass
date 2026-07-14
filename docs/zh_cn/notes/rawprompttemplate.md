# 基于RawPromptTemplate模板构建输入信息

我们新增了 `RawPromptTemplate` 模板解析类，以更好地适配 OpenAI 对话格式。相比此前主要使用的 `PromptTemplate` 类，该类设计更直观，输入配置与最终生成的 message 之间不再存在映射转换，同时支持在模型配置中追加额外的 Prompt 内容。`RawPromptTemplate` 的具体使用方式详见下文。

## 修改数据集的推理配置

基于PromptTemplate的原推理配置格式示例如下：

```python
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt="You are a helpful assistant.",
                )
           ],
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}\nRemember to put your final answer within \\boxed{}.',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

进行如下修改，以调整为RawPromptTemplate的格式：

```python
infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '{problem}\nRemember to put your final answer within \\boxed{}.'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

我们已为常用的客观数据集，例如[AIME2026](opencompass/configs/datasets/aime2026/aime2026_cascade_eval_rawprompt_gen_0970dd.py)，以及[MMLU Pro](opencompass/configs/datasets/mmlu_pro/mmlu_pro_0shot_nocot_genericllmeval_rawprompt_gen_0321fb.py)等，添加了文件名称中包含`rawprompt`的新配置。主观及多轮对话相关数据集也将在近期支持。

## 在模型配置侧添加额外信息

使用`OpenAI`、`OpenAISDK`、`OpenAISDKStreaming`等API对话模型类进行评测时，可通过模型配置中的`meta_template`来附加所需要的额外信息，格式如下：

```python
dict(
    abbr='YOUR_MODEL',
    type=OpenAISDK,
    path='YOUR_MODEL',
    key='YOUR_API_KEY',
    openai_api_base='YOUR_API_BASE',
    meta_template=[
        {'content': 'Your extra system prompt here.', 'role': 'system'},
        {'content': 'Your extra user prompt here.', 'role': 'user'},
    ],
    query_per_second=1,
    batch_size=8,
    temperature=1.0,
    max_out_len=32768,
    max_seq_len=32768,
    ...
)
```
