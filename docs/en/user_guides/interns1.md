# Tutorial for Evaluating Intern-S1

OpenCompass now provides the necessary configs for evaluating Intern-S1. Please perform the following steps to initiate the evaluation of Intern-S1.

## Model Download and Deployment

The Intern-S1 now has been open-sourced, which can be downloaded from [Huggingface](https://huggingface.co/internlm/Intern-S1).
After completing the model download, it is recommended to deploy it as an API service for calling.
You can deploy it based on LMdeploy/vlLM/sglang according to [this page](https://github.com/InternLM/Intern-S1/blob/main/README.md#Serving).

## Evaluation Configs

### Model Configs

We provide a config example in `opencompass/configs/models/interns1/intern_s1.py`.
Please make the changes according to your needs.

```python
models = [
    dict(
        abbr="intern-s1",
        key="YOUR_API_KEY", # Fill in your API KEY here
        openai_api_base="YOUR_API_BASE", # Fill in your API BASE here
        type=OpenAISDK,
        path="internlm/Intern-S1",
        temperature=0.7,
        meta_template=api_meta_template,
        query_per_second=1,
        batch_size=8,
        max_out_len=64000,
        max_seq_len=65536,
        openai_extra_kwargs={
            'top_p': 0.95,
        },
        retry=10,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True} # Control the thinking mode when deploying the model based on vllm or sglang
        },
        pred_postprocessor=dict(type=extract_non_reasoning_content), # Extract non-reasoning contents when opening the thinking mode
    ),
]
```

### Dataset Configs

We provide a config for datasets used for evaluating Intern-S1 in `examples/eval_bench_intern_s1.py`.
You can also add other datasets as needed.

In addition, you need to add the configuration of the LLM Judger in this config file, as shown in the following example:

```python
judge_cfg = dict(
    abbr='YOUR_JUDGE_MODEL',
    type=OpenAISDK,
    path='YOUR_JUDGE_MODEL_PATH',
    key='YOUR_API_KEY',
    openai_api_base='YOUR_API_BASE',
    meta_template=dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ]),
    query_per_second=1,
    batch_size=1,
    temperature=0.001,
    max_out_len=8192,
    max_seq_len=32768,
    mode='mid',
)
```

## Start Evaluation

After completing the above configuration,
enter the following command to start the evaluation:

```bash
opencompass examples/eval_bench_intern_s1.py
```
