# Building Input Messages with RawPromptTemplate

We have introduced the `RawPromptTemplate` class to better align with the OpenAI chat format. Compared to the previously used `PromptTemplate` class, this class features a more intuitive design—there is no mapping transformation between the input configuration and the final generated messages. Additionally, it supports appending extra prompt content in the model configuration. See below for detailed usage of `RawPromptTemplate`.

## Modifying the Dataset Inference Configuration

The original inference configuration format based on PromptTemplate is shown below:

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

Modify it as follows to switch to the RawPromptTemplate format:

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

We have added new configurations with `rawprompt` in their filenames for commonly used objective datasets, such as [AIME2026](opencompass/configs/datasets/aime2026/aime2026_cascade_eval_rawprompt_gen_0970dd.py) and [MMLU Pro](opencompass/configs/datasets/mmlu_pro/mmlu_pro_0shot_nocot_genericllmeval_rawprompt_gen_0321fb.py). Support for subjective and multi-turn conversation datasets will be added soon.

## Adding Extra Information in Model Configuration

When using API-based chat model classes such as `OpenAI`, `OpenAISDK`, or `OpenAISDKStreaming` for evaluation, you can append extra information via the `meta_template` in the model configuration. The format is as follows:

```python
dict(
    abbr='YOUR_MODEL',
    type=OpenAISDK,
    path='YOUR_MODEL',
    key='YOUR_API_KEY',
    openai_api_base='YOUR_API_BASE',
    meta_template=[
        {'content': 'Your extra system prompt prompt here.', 'role': 'system'},
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
