# 官网学术榜单精度复现指引

为快捷、直观地向用户展示主流开源社区模型及商用模型在常用数据集上的综合表现，我们在官网以通常两周更新一次的频率持续维护大语言模型的[学术榜单](https://rank.opencompass.org.cn/leaderboard-llm-academic/?m=REALTIME)。

由于模型和数据集的迭代以及OpenCompass评测平台的持续建设，学术榜单的评测配置可能会不断变化，具体而言，我们遵循以下更新规则：

- 加入新发布模型的同时，已发布半年到一年或以上的模型将从榜单中移除。
- 加入新数据集的同时，精度接近饱和的数据集将被移除。
- 根据评测配置的变化，同步更新榜单上原有的评测结果。

在OpenCompass项目中提供了学术榜单所使用的实时配置文件，以支持快速复现。

## 学术榜单评测复现

[eval_academic_leaderboard_REALTIME.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_academic_leaderboard_REALTIME.py) 中包含了目前学术榜单评测所使用的配置，你可以通过顺序执行以下步骤来完成评测复现。

### 1: 模型配置

首先，修改[eval_academic_leaderboard_REALTIME.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_academic_leaderboard_REALTIME.py)中的Model List代码块中加入你希望评测的模型。

```python
# Models (add your models here)
from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import \
    models as hf_internlm2_5_7b_chat_model
```

原有示例中调用了一个OpenCompass中已集成的lmdeploy型模型配置文件，你也可以基于[此文档](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/models.html)自行定义新的模型配置。一个基于OpenAISDK调用已部署服务评测Qwen3-235B-A22B的配置示例如下：

```python
from opencompass.models import OpenAISDK
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

qwen3_235b_a22b_model = dict(
    abbr="qwen_3_235b_a22b_thinking", # 用于保存结果时标识该模型配置
    key="YOUR_SERVE_API_KEY",
    openai_api_base="YOUR_SERVE_API_URL",
    type=OpenAISDK, # 模型配置类型，常用如OpenAISDK、TurboMindModelwithChatTemplate、HuggingFacewithChatTemplate
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
    }, # 模型的额外配置内容，例如qwen3中控制其是否思考的选项
    pred_postprocessor=dict(type=extract_non_reasoning_content), # 对于输出带有think tag的推理模型，添加此pred_postprocessor可以提取其think后的输出
)

models = [
    qwen3_235b_a22b_model,
]
```

学术榜单中部分常见参数的设置如下所示。

- `max_seq_len` = 65536 or 32768
- `max_out_len` = 64000 or 32000
- `temperature` = 0.6
- `top_p` = 0.95

### 2: Verifier配置

接着，在 `judge_cfg` 中补全你的Verifier模型信息。有关于LLM Verifier的详细内容，请参阅[此文档](https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/llm_judge.html)。
目前，学术榜单使用[CompassVerifier-32B](https://huggingface.co/opencompass/CompassVerifier-32B)，基于OpenAISDK的配置示例如下：

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

### 3: 执行评测

完善上述配置文件后即可在命令行中输入如下内容，开始执行评测：

```bash
  opencompass examples/eval_academic_leaderboard_REALTIME.py
```

有关更加详细的命令行评测参数，请参阅[此文档](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/experimentation.html)。
