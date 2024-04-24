# 评测 LMDeploy 模型

我们支持评测使用 [LMDeploy](https://github.com/InternLM/lmdeploy) 加速过的大语言模型。LMDeploy 由 MMDeploy 和 MMRazor 团队联合开发，是涵盖了 LLM 任务的全套轻量化、部署和服务解决方案。 **TurboMind** 是 LMDeploy 推出的高效推理引擎。OpenCompass 对 TurboMind 进行了适配，本教程将介绍如何使用 OpenCompass 来对 TurboMind 加速后的模型进行评测。

## 环境配置

### 安装 OpenCompass

请根据 OpenCompass [安装指南](https://opencompass.readthedocs.io/en/latest/get_started/installation.html) 来安装算法库和准备数据集。

### 安装 LMDeploy

使用 pip 安装 LMDeploy (python 3.8+)：

```shell
pip install lmdeploy
```

## 评测

OpenCompass 支持分别通过 turbomind python API 评测数据集。

下文以 InternLM-20B 模型为例，介绍如何评测。首先我们准备好测试配置文件`configs/eval_internlm_turbomind.py`:

```python
from mmengine.config import read_base
from opencompass.models.turbomind import TurboMindModel


with read_base():
    # choose a list of datasets
    from .datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from .datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import WiC_datasets
    from .datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # and output the results in a chosen format
    from .summarizers.medium import summarizer

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# config for internlm-20b model
internlm_20b = dict(
        type=TurboMindModel,
        abbr='internlm-20b-turbomind',
        path="internlm/internlm-20b", # 注意路径与huggingface保持一致
        engine_config=dict(session_len=2048,
                           max_batch_size=8,
                           rope_scaling_factor=1.0),
        gen_config=dict(top_k=1, top_p=0.8,
                        temperature=1.0,
                        max_new_tokens=100),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        concurrency=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<eoa>'
    )

models = [internlm_20b]
```

然后，在 OpenCompass 的项目目录下，执行如下命令可得到评测结果：

```shell
python run.py configs/eval_internlm_turbomind.py -w outputs/turbomind/internlm-20b
```

**注：**

- 如果想在测评配置文件中`engine_config`和`gen_config`字段传递更多参数，请参考[TurbomindEngineConfig](https://lmdeploy.readthedocs.io/zh-cn/latest/inference/pipeline.html#turbomindengineconfig) 和 [EngineGenerationConfig](https://lmdeploy.readthedocs.io/zh-cn/latest/inference/pipeline.html#generationconfig)
- 如果评测 InternLM Chat 模型，请使用配置文件 `eval_internlm_chat_turbomind.py`
- 如果评测 InternLM 7B 模型，请修改 `eval_internlm_turbomind.py` 或者 `eval_internlm_chat_turbomind.py`。将`models`字段配置为`models = [internlm_7b]` 。
