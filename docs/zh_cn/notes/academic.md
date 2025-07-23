# 官网学术榜单精度复现指引

为快捷、直观地向用户展示主流开源社区模型及商用模型在常用数据集上的综合表现，我们在官网以通常两周更新一次的频率持续维护大语言模型的[学术榜单](https://rank.opencompass.org.cn/leaderboard-llm-academic/?m=REALTIME)。

由于模型和数据集的迭代以及OpenCompass评测平台的持续建设，学术榜单的评测配置可能会不断变化，具体而言，我们遵循以下更新规则：

- 加入新发布模型的同时，已发布半年到一年或以上的模型将从榜单中移除。
- 加入新数据集的同时，精度接近饱和的数据集将被移除。
- 根据评测配置的变化，同步更新榜单上原有的评测结果。

在OpenCompass项目中提供了学术榜单所使用的实时配置文件，以支持快速复现。

## 学术榜单评测复现

[eval_academic_leaderboard_REALTIME.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_academic_leaderboard_REALTIME.py) 中包含了目前学术榜单评测所使用的配置，请运行该配置文件以进行复现，并补全以下信息。

### 模型配置

在Model List代码块中加入你希望评测的模型。

```python
# Models (add your models here)
from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import \
    models as hf_internlm2_5_7b_chat_model
```

此处提供了目前我们常用的模型配置参数以供参考。

- `max_seq_len` = 65536
- `max_out_len` = 64000
- `temperature` = 0.6
- `top_p` = 0.95
- `top_k` = 20

### Verifier配置

在 `judge_cfg` 中补全你的Verifier模型信息。目前，学术榜单使用[CompassVerifier-32B](https://huggingface.co/opencompass/CompassVerifier-32B)，基于OpenAISDK的配置示例如下：

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
