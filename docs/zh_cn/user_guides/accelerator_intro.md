# 使用 vLLM 或 LMDeploy 来一键式加速评测推理

## 背景

在 OpenCompass 评测过程中，默认使用 Hugging Face 的 transformers 库进行推理，这是一个非常通用的方案，但在某些情况下，我们可能需要更高效的推理方法来加速这一过程，比如借助 vLLM 或 LMDeploy。

- [LMDeploy](https://github.com/InternLM/lmdeploy) 是一个用于压缩、部署和服务大型语言模型（LLM）的工具包，由 [MMRazor](https://github.com/open-mmlab/mmrazor) 和 [MMDeploy](https://github.com/open-mmlab/mmdeploy) 团队开发。
- [vLLM](https://github.com/vllm-project/vllm) 是一个快速且易于使用的 LLM 推理和服务库，具有先进的服务吞吐量、高效的 PagedAttention 内存管理、连续批处理请求、CUDA/HIP 图的快速模型执行、量化技术（如 GPTQ、AWQ、SqueezeLLM、FP8 KV Cache）以及优化的 CUDA 内核。

本章以 Qwen3.5-35B-A3B 为例——它同时兼容 vLLM 和 LMDeploy。

## 加速前准备

首先，请检查您要评测的模型是否支持使用 vLLM 或 LMDeploy 进行推理加速。其次，请确保您已经安装了 vLLM 或 LMDeploy，具体安装方法请参考它们的官方文档，也可以直接通过 pip 安装：

```bash
pip install lmdeploy   # LMDeploy（Python 3.8+）
pip install vllm       # vLLM
```

## 评测时使用 vLLM 或 LMDeploy

### 方法1：使用命令行参数来变更推理后端

OpenCompass 提供了一键式的评测加速，可以在评测过程中自动将 Hugging Face transformers 模型转换为 vLLM 或 LMDeploy 模型。以下是使用 Qwen3.5-35B-A3B 评测 GSM8K 演示数据集的样例配置：

```python
# eval_gsm8k.py
from mmengine.config import read_base

with read_base():
    # 选择一个数据集列表
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import \
        gsm8k_datasets as datasets
```

模型使用基于 Hugging Face transformers 的 Qwen3.5-35B-A3B 配置：

```python
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

默认 Hugging Face 版本的运行方式如下：

```bash
opencompass eval_gsm8k.py
```

如果需要使用 vLLM 或 LMDeploy 进行加速评测，只需追加 `-a` 参数：

```bash
opencompass eval_gsm8k.py -a vllm
```

或

```bash
opencompass eval_gsm8k.py -a lmdeploy
```

转换规则：`HuggingFacewithChatTemplate` 会被自动转换为 `VLLMwithChatTemplate` 或 `TurboMindModelwithChatTemplate`，其中张量并行（`tensor_parallel_size` / `tp`）取自 `run_cfg.num_gpus`，`max_model_len` / `session_len` 取自 `max_seq_len`，生成参数尽量保留；LMDeploy 默认贪心解码，需要采样时应设置 `do_sample=True`。

仓库中已经提供 Qwen3.5-35B-A3B 的两种后端配置：`opencompass/configs/models/qwen3/vllm_qwen3_5_35b_a3b.py` 和 `lmdeploy_qwen3_5_35b_a3b.py`。它们包含更完整的参数（如 `enable_thinking`、思维链输出的后处理），正式评测推荐直接引用这些配置，而不是依赖运行时转换。

### 方法2：通过部署推理加速服务 API 来加速评测

OpenCompass 还支持通过部署 vLLM 或 LMDeploy 的推理加速服务 API 来加速评测，参考步骤如下：

1. 安装 openai 包：

```bash
pip install openai
```

2. 部署 vLLM 或 LMDeploy 的推理加速服务 API，具体部署方法请参考它们的官方文档，下面以 LMDeploy 为例：

```bash
lmdeploy serve api_server Qwen/Qwen3.5-35B-A3B \
    --model-name Qwen3.5-35B-A3B \
    --tp 2 \
    --server-port 23333
```

`api_server` 启动时的参数可以通过命令行 `lmdeploy serve api_server -h` 查看。比如 `--tp` 设置张量并行，`--session-len` 设置推理的最大上下文窗口长度，`--cache-max-entry-count` 调整 k/v cache 的内存使用比例等等。

3. 服务部署成功后，修改评测脚本，将模型配置改为指向部署的服务地址，通过 OpenAI 兼容接口调用：

```python
from opencompass.models import OpenAISDK

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [
    dict(
        abbr='Qwen3.5-35B-A3B-LMDeploy-API',
        type=OpenAISDK,
        key='EMPTY',  # API key
        openai_api_base='http://0.0.0.0:23333/v1',  # 服务地址
        path='Qwen3.5-35B-A3B',  # 请求服务时的 model name
        tokenizer_path='Qwen/Qwen3.5-35B-A3B',  # tokenizer 名称或路径，为 None 时使用默认 tokenizer gpt-4
        rpm_verbose=True,  # 是否打印请求速率
        meta_template=api_meta_template,  # 服务请求模板
        query_per_second=1,  # 服务请求速率
        max_out_len=8192,  # 最大输出长度
        max_seq_len=131072,  # 最大输入长度
        temperature=0.6,  # 生成温度
        batch_size=8,  # 批处理大小
        retry=3,  # 重试次数
    )
]
```

## 加速效果及性能对比

下表是一组历史参考数据（Llama-3-8B-Instruct，单卡 A800，GSM8K），说明两种后端相对 Hugging Face transformers 的典型加速量级：

| 推理后端     | 精度（Accuracy） | 推理时间（分钟：秒） | 加速比（相对于 Hugging Face） |
| ------------ | ---------------- | -------------------- | ----------------------------- |
| Hugging Face | 74.22            | 24:26                | 1.0                           |
| LMDeploy     | 73.69            | 11:15                | 2.2                           |
| vLLM         | 72.63            | 07:52                | 3.1                           |

实际的加速比取决于模型结构、显卡型号和数据集；不同后端的采样实现差异也可能小幅影响精度。
