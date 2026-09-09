# 模型接入总览

模型配置回答三个问题：调用哪一种模型后端、如何生成回复、运行一个模型实例需要多少资源。先根据部署形态选择入口，再调整参数。

| 部署形态                  | 推荐入口                                               | 适用场景                       |
| ------------------------- | ------------------------------------------------------ | ------------------------------ |
| 本地 Huggingface 原生加载 | `HuggingFacewithChatTemplate` 或 `HuggingFaceCausalLM` | 小规模验证、兼容性基线         |
| 本地推理引擎加速          | LMDeploy 或 vLLM 模型类                                | 大模型、多卡和批量评测         |
| OpenAI 兼容服务           | `OpenAISDK` 等 API 模型类                              | 商业模型、自建推理服务         |
| 其他厂商API接口           | 对应 API 模型类或 LiteLLM                              | 接口协议、鉴权和限流不同的服务 |
| 仓库尚未支持的后端        | 自定义 `BaseModel` 子类                                | 特殊输入输出或私有运行时       |

本地权重与接口模型的具体配置见下文[本地权重模型](#本地权重模型)与[接口与推理服务模型](#接口与推理服务模型)两节；自定义后端参阅[新增模型后端](../extension/new_model.md)。

## 所有模型都应关注的字段

```python
models = [
    dict(
        type=...,                 # 模型实现
        abbr='model-name',        # 结果目录和表格中的稳定简称
        path='model-or-service',  # 权重路径、Hub ID 或服务模型名
        max_seq_len=32768,        # 模型允许的总上下文长度
        max_out_len=4096,         # 单次最大生成长度
        batch_size=8,
    )
]
```

`max_seq_len` 与 `max_out_len` 不是越大越好：二者共同影响可容纳的输入、显存或服务费用。`run_cfg` 是调度资源声明，不等同于模型内部的张量并行配置；使用 LMDeploy/vLLM 时必须同时核对两者。

## 模型配置与数据集配置的边界

模型配置负责模型协议、生成参数、资源和接口行为；数据集配置负责题目内容、Prompt 和评分方式。模型专属特殊 token 应由 tokenizer chat template 或 [MetaTemplate](../prompt/meta_template.md) 管理，通用题目指令优先放在数据集侧的 [RawPromptTemplate](../prompt/raw_prompt_template.md)。

## 选择现有配置

```bash
python tools/list_configs.py qwen vllm
```

打开命中的配置并核对模型版本、依赖、上下文长度和生成参数。目录中存在配置只表示已提供接入样例，不保证它与当前服务版本、显卡或推理框架版本兼容。

## 本地权重模型

### Transformers 对话模型

对支持 Hugging Face chat template 的指令模型，优先使用 `HuggingFacewithChatTemplate`：

```python
from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2-1.5b-instruct-hf',
        path='Qwen/Qwen2-1.5B-Instruct',
        model_kwargs=dict(device_map='auto'),
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        generation_kwargs=dict(do_sample=False),
        max_seq_len=32768,
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
```

`path` 可以是 Hub ID，也可以是本地权重目录。需要执行模型仓库自定义代码时才启用 `trust_remote_code=True`，并应固定可信 revision。

### 基座模型与 PPL 评测

基座模型或需要 token 概率的判别式任务可使用 `HuggingFaceCausalLM`。对话模型通常只使用生成式配置；不要用 PPL 配置评价只暴露聊天接口、无法返回 token 概率的模型。

### 长度、批处理与显存

- `max_seq_len` 应覆盖输入和生成预算；截断会改变评测题目。
- `max_out_len` 过小会截断答案，过大会降低吞吐并增加费用。
- `batch_size` 是 OpenCompass 送入模型的批大小，不保证后端实际按同一方式批处理。
- 左填充通常适合 decoder-only 批量生成；是否支持 padding 仍以模型实现为准。
- `run_cfg.num_gpus` 告诉 Runner 一个任务占几张卡，不会自动让任意模型实现张量并行。

先对演示数据集执行 `--dry-run`，再用少量样本检查 prompt、截断、停止词和生成结果。显存不足时优先减小批量和上下文预算，再考虑量化或高吞吐后端。

### 多卡与并行

单个模型实例拆到多张卡（张量并行）时，并行度在模型配置内部设置，占卡数由 `run_cfg` 向调度声明，两者必须核对一致：

```python
models = [
    dict(
        type=...,                                   # 如 VLLMwithChatTemplate
        abbr='my-model-tp2',
        path='org/model',
        model_kwargs=dict(tensor_parallel_size=2),  # 后端内部张量并行
        run_cfg=dict(num_gpus=2),                   # 向调度声明的占卡数
    )
]
```

API 模型 `run_cfg.num_gpus=0`，不占本地 GPU。

### 加速后端

CLI 的 `--accelerator vllm` 或 `--accelerator lmdeploy` 可以转换部分 Hugging Face 模型配置，但只覆盖受支持的模型类型和参数。正式评测更推荐审阅并固定对应的 vLLM/LMDeploy 配置，详见[推理后端](accelerator_intro.md)。

## 接口与推理服务模型

API 模型不占用本地 GPU，但受鉴权、限流、网络、服务端模型版本和费用影响。OpenCompass 提供 OpenAI 及 OpenAI SDK 兼容实现，也包含若干厂商适配器；应优先复用最接近目标协议的现有配置。

### OpenAI 兼容配置

仓库现有 OpenAI 配置使用 `key='ENV'` 通过手动更改或从环境变量读取密钥：

```python
import os
from opencompass.models import OpenAI

models = [
    dict(
        type=OpenAI,
        abbr='my-api-model',
        path='served-model-name',
        key='ENV',
        openai_api_base='OPENAI_API_BASE',
        query_per_second=1,
        max_seq_len=32768,
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=0),
    )
]
```

密钥不要写入提交到 Git 的配置。不同模型类读取的环境变量和 endpoint 字段可能不同，使用前应检查对应类与仓库示例。

### 并发、重试与确定性

`batch_size`、`query_per_second` 和 Runner worker 数会共同影响请求并发。出现 429、超时或服务端错误时，应先按服务配额降低并发，再调整重试；无限重试可能造成重复费用。

即使 `temperature=0`，服务端升级、路由、推理实现和安全策略也可能改变回复。正式结果应记录：

- 服务模型的精确名称或版本；
- API base、模型供应方和区域；
- 请求时间、生成参数和重试策略；
- OpenCompass 配置快照及异常请求数量。

### 输入协议

新数据集优先使用 [RawPromptTemplate](../prompt/raw_prompt_template.md) 产生 `role/content` 消息。若模型配置还会通过 `meta_template` 追加 system/user 消息，必须预览最终输入，避免重复指令。推理前用极小数据集验证鉴权、消息格式、停止条件和错误处理，再扩大并发。
