# 本地模型的一站式部署和评测

本页介绍在同一次 `opencompass` 运行中完成本地模型加载、推理和评测的方法。OpenCompass 可以把 Hugging Face 模型配置转换为 vLLM 或 LMDeploy 后端；模型由评测进程直接管理，不需要预先启动独立推理服务。

如果希望先启动共享服务，再让评测进程通过 API 调用，请改用[模型接入总览中的服务化部署方案](../user_guides/models.md#通过部署推理加速服务来运行)。

## 准备推理后端

[LMDeploy](https://github.com/InternLM/lmdeploy) 和 [vLLM](https://github.com/vllm-project/vllm) 都能以高吞吐推理引擎加载本地权重。先确认目标模型受对应框架支持，再按框架的安装文档准备环境，也可以直接安装：

```bash
pip install lmdeploy
pip install vllm
```

## 使用命令行切换后端

下面以 Qwen3.5-35B-A3B 和 GSM8K 演示数据集为例，新建 `eval_gsm8k.py`：

```python
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import \
        gsm8k_datasets as datasets

from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen3.5-35b-a3b-hf',
        path='Qwen/Qwen3.5-35B-A3B',
        max_seq_len=262144,
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
    )
]
```

默认使用 Hugging Face transformers 后端：

```bash
opencompass eval_gsm8k.py
```

追加 `-a` 或 `--accelerator` 即可在本次运行中切换后端：

```bash
opencompass eval_gsm8k.py -a vllm
opencompass eval_gsm8k.py -a lmdeploy
```

转换时，`HuggingFacewithChatTemplate` 会变为 `VLLMwithChatTemplate` 或 `TurboMindModelwithChatTemplate`：

- `tensor_parallel_size` / `tp` 取自 `run_cfg.num_gpus`；
- `max_model_len` / `session_len` 取自 `max_seq_len`；
- 生成参数会尽量保留；
- LMDeploy 默认使用贪心解码，需要采样时应设置 `do_sample=True`。

CLI 转换只覆盖部分模型类型和参数。仓库已提供 `configs/models/qwen3/vllm_qwen3_5_35b_a3b.py` 与 `configs/models/qwen3/lmdeploy_qwen3_5_35b_a3b.py`，其中包含 `enable_thinking`、思维链后处理等完整参数；正式评测优先审阅并直接引用对应后端配置。

## 加速效果参考

下表是一组历史参考数据（Llama-3-8B-Instruct，单卡 A800，GSM8K）：

| 推理后端     | 精度（Accuracy） | 推理时间（分钟：秒） | 相对 Hugging Face 加速比 |
| ------------ | ---------------: | -------------------: | -----------------------: |
| Hugging Face |            74.22 |                24:26 |                     1.0× |
| LMDeploy     |            73.69 |                11:15 |                     2.2× |
| vLLM         |            72.63 |                07:52 |                     3.1× |

实际加速比取决于模型结构、显卡型号、数据集、上下文长度和采样配置；不同后端的采样实现差异也可能影响精度。

需要把模型作为独立服务供多人共享时，参阅[通过部署推理加速服务来运行](../user_guides/models.md#通过部署推理加速服务来运行)。
