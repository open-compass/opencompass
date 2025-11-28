# Guide to Reproducing CompassAcademic Leaderboard Results

To provide users with a quick and intuitive overview of the performance of mainstream open-source and commercial models on widely-used datasets, we maintain the [CompassAcademic Leaderboard](https://rank.opencompass.org.cn/leaderboard-llm-academic/?m=REALTIME) for LLMs on our official website, updating it typically every two weeks.

Given the continuous iteration of models and datasets, along with ongoing upgrades to the OpenCompass, the configuration settings for the CompassAcademic leaderboard may evolve. Specifically, we adhere to the following update principles:

- Newly released models are promptly included, while models published six months to one year (or more) ago are removed from the leaderboard.
- New datasets are incorporated, while datasets nearing performance saturation are phased out.
- Existing evaluation results on the leaderboard are updated in sync with changes to the evaluation configuration.

To support rapid reproducibility, OpenCompass provides the real-time configuration files used in the academic leaderboard.

## CompassAcademic Leaderboard Reproduction

[eval_academic_leaderboard_REALTIME.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_academic_leaderboard_REALTIME.py) contains the configuration currently used for academic ranking evaluation. You can replicate the evaluation by following the steps as follows.

### 1: Model Configs

Firstly, modify the Model List code block in [eval_academic_leaderboard_REALTIME.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_academic_leaderboard_REALTIME.py) to include the model you wish to evaluate.

```python
# Models (add your models here)
from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import \
    models as hf_internlm2_5_7b_chat_model
```

The original example calls an lmdeploy-based model configuration in OpenCompass.
You can also build your new model configuration based on [this document](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/models.html).
An example of a configuration that calls the deployed service of Qwen3-235B-A22B based on OpenAISDK is as follows:

```python
from opencompass.models import OpenAISDK
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

qwen3_235b_a22b_model = dict(
    abbr="qwen_3_235b_a22b_thinking", # Used to identify the model configuration
    key="YOUR_SERVE_API_KEY",
    openai_api_base="YOUR_SERVE_API_URL",
    type=OpenAISDK, # The model configuration types, commonly used such as OpenAISDK, TurboMindModelwithChatTemplate, HuggingFacewithChatTemplate
    path="Qwen/Qwen3-235B-A22B",
    temperature=0.6,
    meta_template=dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
),
    query_per_second=1,
    max_out_len=32000,
    max_seq_len=32768,
    batch_size=8,
    retry=10,
    extra_body={
        'chat_template_kwargs': {'enable_thinking': True},
    }, # Additional configurations of the model, such as the option in Qwen3 series to control whether they thinks or not
    pred_postprocessor=dict(type=extract_non_reasoning_content), # adding this pred_postprocessor can extract the non-reasoning content from models that output with a think tag
)

models = [
    qwen3_235b_a22b_model,
]
```

Here are the commonly used parameters for reference.

- `max_seq_len` = 65536 or 32768
- `max_out_len` = 64000 or 32000
- `temperature` = 0.6
- `top_p` = 0.95

### 2: Verifier Configs

Complete your verifier model information in `judge_cfg`.
For detailed information about LLM verifiers, please refer to [this document](https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/llm_judge.html).
At present, CompassAcademic use [CompassVerifier-32B](https://huggingface.co/opencompass/CompassVerifier-32B), here is the config example using OpenAISDK:

```python
judge_cfg = dict(
    abbr='CompassVerifier',
    type=OpenAISDK,
    path='opencompass/CompassVerifier-32B',
    key='YOUR_API_KEY',
    openai_api_base='YOUR_API_BASE',
    meta_template=dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ]),
    query_per_second=1,
    batch_size=8,
    temperature=0.001,
    max_out_len=8192,
    max_seq_len=32768,
    mode='mid',
)
```

### 3: Execute evaluation

After completing the above configuration file, you can enter the following content in the CLI to start the evaluation:

```bash
  opencompass examples/eval_academic_leaderboard_REALTIME.py
```

For more detailed CLI parameters, please refer to [this document](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/experimentation.html)。
