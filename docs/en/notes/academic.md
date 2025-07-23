# Guide to Reproducing CompassAcademic Leaderboard Results

To provide users with a quick and intuitive overview of the performance of mainstream open-source and commercial models on widely-used datasets, we maintain the [CompassAcademic Leaderboard](https://rank.opencompass.org.cn/leaderboard-llm-academic/?m=REALTIME) for LLMs on our official website, updating it typically every two weeks.

Given the continuous iteration of models and datasets, along with ongoing upgrades to the OpenCompass, the configuration settings for the CompassAcademic leaderboard may evolve. Specifically, we adhere to the following update principles:

- Newly released models are promptly included, while models published six months to one year (or more) ago are removed from the leaderboard.
- New datasets are incorporated, while datasets nearing performance saturation are phased out.
- Existing evaluation results on the leaderboard are updated in sync with changes to the evaluation configuration.

To support rapid reproducibility, OpenCompass provides the real-time configuration files used in the academic leaderboard.

## CompassAcademic Leaderboard Reproduction

[eval_academic_leaderboard_REALTIME.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_academic_leaderboard_REALTIME.py) contains the configuration currently used for academic ranking evaluation. Please run the configuration file to reproduce and complete the following information:

### Model Configs

Add the model you want to evaluate to the Model List code block.

```python
# Models (add your models here)
from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import \
    models as hf_internlm2_5_7b_chat_model
```

Here are the commonly used parameters for reference.

- `max_seq_len` = 65536
- `max_out_len` = 64000
- `temperature` = 0.6
- `top_p` = 0.95
- `top_k` = 20

### Verifier Configs

Complete your verifier model information in `judge_cfg`. At present, CompassAcademic use [CompassVerifier-32B](https://huggingface.co/opencompass/CompassVerifier-32B), here is the config example using OpenAISDK:

```python
dict(
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
