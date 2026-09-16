# 模型配置总览

OpenCompass 的一个标准模型配置将包含下面的信息：模型类别、上下文等模型超参、推理并发数、部署所需资源等等。主要有以下几个模型类别。


| 部署形态                   | 推荐入口                                               | 适用场景                                |
| -------------------------- | ------------------------------------------------------ | --------------------------------------- |
| OpenAI 兼容服务            | `OpenAISDK`、`OpenAISDKStreaming`、`OpenAISDKResponse`                       | OpenAI 官方接口、自建 OpenAI 格式推理服务、中转网关 |
| 其它厂商 API                   | `GeminiSDK`、`ClaudeSDK` 等模型类                  | 使用厂商原生协议、鉴权及参数传入        |
| 本地推理引擎加速           | `TurboMindModelwithChatTemplate`（LMDeploy）及 `VLLMwithChatTemplate`（vLLM）等模型类                                | 一站式部署与评测                  |
| 本地 Hugging Face 原生加载 | `HuggingFacewithChatTemplate` 或 `HuggingFaceCausalLM` | 一站式部署与评测                  |


下文分别介绍通过 API 调用模型，以及在 OpenCompass 进程中一站式加载本地权重并完成评测的配置方法；自定义后端参阅[新增模型后端](../extension/new_model.md)。

## 1. API 模型

API 模型由远端服务完成推理，OpenCompass 负责编排请求并保存结果。配置前应确认服务地址、模型名称、密钥、请求限额和超时设置。API 模型本身通常不占用评测机 GPU，可将 `run_cfg.num_gpus` 设为 `0`。

### 共通参数

常用字段如下。不同模型类支持的参数并不完全相同，应以对应类的构造函数和仓库中的现有配置为准。

| 参数 | 说明 |
| ---- | ---- |
| `type` | OpenCompass 使用的模型类，例如 `OpenAISDKResponse`。 |
| `abbr` | 模型简称，用于输出目录、结果文件和汇总表。 |
| `path` | 服务端模型名称；部分模型类使用其他字段，例如 `TurboMindAPIModel` 使用 `model_name`。 |
| `key` | API 密钥。可直接传入字符串，也可通过 `os.getenv()` 读取自定义环境变量。 |
| `max_seq_len` | 模型允许的最大序列长度，输入与输出之和不应超过该值。 |
| `max_out_len` | 单次请求允许生成的最大 token 数。数据集侧 Inferencer 显式设置同名参数时，以数据集配置为准。 |
| `temperature` | 采样温度；可用范围和实际语义以服务端实现为准。 |
| `query_per_second` | 每秒请求数上限。设置过高可能触发服务端限流。 |
| `batch_size` | `Inferencer` 推理时的批处理大小。评测使用 `OpenICLInferTask` 时，等效于 API 并发线程数。 |
| `max_workers` | API 请求的最大并发线程数，仅适用于声明了该参数的模型类。评测使用 `OpenICLInferConcurrentTask` 时，`max_workers` 成为控制此模型并发的唯一参数。 |
| `retry` | 请求失败后的最大重试次数。 |
| `tokenizer_path` | 估算输入长度的 tokenizer 名称或路径。使用中转服务或自定义模型名时通常需要显式指定。 |

### OpenAISDK：Chat Completions

`OpenAISDK` 通过 `client.chat.completions.create` 调用 Chat Completions 接口，适用于 OpenAI 官方服务以及实现了相同接口的兼容服务：

```python
from opencompass.models import OpenAISDK

models = [
    dict(
        type=OpenAISDK,
        abbr='openai-compatible-model',
        path='served-model-name',
        key='ENV',  # OPENAI_API_KEY
        openai_api_base='https://example.com/v1',
        tokenizer_path='org/model-tokenizer',
        max_seq_len=131072,
        max_out_len=131072,
        query_per_second=1,
        max_workers=8,
        timeout=3600,
        retry=10,
        status_code_mappings={
            400: 'The request was rejected by the service.',
        },
        openai_extra_kwargs=dict(top_p=0.95),
        extra_body=dict(
            chat_template_kwargs=dict(enable_thinking=True),
        ),
        batch_size=8,
    )
]
```

主要参数说明如下：

- `openai_extra_kwargs`：此参数中的字段会直接加入 Chat Completions 请求体。
- `extra_body`：用于传递兼容服务开放的额外请求体字段。
- `status_code_mappings`：将指定的 HTTP 错误状态码映射为固定的模型输出。例如，上述配置在服务返回 400 时直接返回映射文本，而不再重试该请求；未配置映射的错误仍按 `retry` 设置进行重试。该参数适合为内容过滤等可预期的服务端拒绝提供占位输出，不应将鉴权失败、限流或服务异常映射为正常输出。

### OpenAISDKStreaming：流式 Chat Completions

`OpenAISDKStreaming` 继承自 `OpenAISDK`，使用 Chat Completions 的流式响应，并在流结束后将完整文本交给评测流程。它适合服务端要求流式调用或生成时间较长的场景：

```python
import os

from opencompass.models import OpenAISDKStreaming

models = [
    dict(
        type=OpenAISDKStreaming,
        abbr='openai-compatible-streaming',
        path='served-model-name',
        key='ENV',  # OPENAI_API_KEY
        openai_api_base='https://example.com/v1',
        tokenizer_path='org/model-tokenizer',
        max_seq_len=131072,
        max_out_len=131072,
        query_per_second=1,
        max_workers=8,
        timeout=3600,
        retry=10,
        stream=True,
        finish_reason_confirm=True,
        verbose=True,
        batch_size=8,
    )
]
```

除 `OpenAISDK` 的参数外，流式模型还提供以下参数：

- `finish_reason_confirm`：是否要求流的最终响应包含 `finish_reason`，默认为 `True`。若流结束时仍未收到该字段，当前响应会被视为不完整并触发重试；设为 `False` 时则返回已经收集到的文本。
- `verbose`：设为 `True` 时，除记录请求开始、结束原因和耗时等日志外，还会将每个流式响应块中的思考内容与回答文本实时输出到终端。相比之下，普通 `OpenAISDK` 的 `verbose` 仅记录请求及响应处理日志，不会逐块输出生成内容。该选项适合调试长时间生成；并发请求较多时，不同请求的终端输出可能交错。

### OpenAISDKResponse：Responses API

`OpenAISDKResponse` 调用 OpenAI Responses API，即 `client.responses.create`。Responses API 专属参数通过 `openai_extra_kwargs` 传入，具体字段可参阅 [OpenAI Responses API 文档](https://developers.openai.com/api/reference/resources/responses/methods/create/)。配置示例如下：

```python
from opencompass.models import OpenAISDKResponse

models = [
    dict(
        type=OpenAISDKResponse,
        abbr='gpt-6-astra-response',
        path='gpt-6-astra',
        key='ENV',  # OPENAI_API_KEY
        openai_api_base='https://api.openai.com/v1',
        max_seq_len=131072,
        max_out_len=131072,
        query_per_second=1,
        max_workers=8,
        timeout=3600,
        retry=10,
        openai_extra_kwargs=dict(reasoning=dict(effort='max')),
        batch_size=8,
    )
]
```

### 其他厂商 API

厂商原生 SDK 的鉴权方式、模型名称和生成参数差异较大，OpenCompass 仅负责将统一的评测输入转换为相应协议。使用前应查阅对应厂商提供的官方接口文档，并核对当前 SDK 版本与 API 账号权限。下面提供部分主流厂商 SDK 的配置实例：

Gemini SDK 示例：

```python
import os

from opencompass.models import GeminiSDK

models = [
    dict(
        type=GeminiSDK,
        abbr='gemini-3.1-pro-preview',
        path='gemini-3.1-pro-preview',
        key='GEMINI_API_KEY',
        max_seq_len=1000000,
        max_out_len=65536,
        query_per_second=1,
        max_workers=8,
        retry=10,
        temperature=1.0,
        thinking=dict(thinking_level='high', include_thoughts=True),
        batch_size=8,
    )
]
```

Claude SDK 示例：

```python
import os

from opencompass.models import ClaudeSDK

models = [
    dict(
        type=ClaudeSDK,
        abbr='claude-opus-5',
        path='claude-opus-5',
        key='ANTHROPIC_API_KEY',
        max_seq_len=1000000,
        max_out_len=131072,
        query_per_second=1,
        max_workers=8,
        retry=10,
        temperature=1.0,
        thinking=dict(type='adaptive'),
        claude_extra_kwargs=dict(
            output_config=dict(effort='max'),
        ),
        batch_size=8,
    )
]
```

### 本地部署推理服务来进行评测

如果拥有本地 GPU 资源，并希望使用 OpenCompass 评测本地部署的模型，建议将模型部署与评测流程解耦：先通过 LMDeploy 等推理后端启动独立服务，再由 OpenCompass 的 API 模型类调用该服务。以下示例使用 LMDeploy 部署模型：

```bash
lmdeploy serve api_server Qwen/Qwen3.5-35B-A3B \
    --model-name Qwen3.5-35B-A3B \
    --tp 2 \
    --server-port 23333
```

确认服务可以正常访问后，根据服务提供的接口和评测方式选择相应的模型类。对于兼容 OpenAI 格式的生成接口，可以使用 `OpenAISDK` 或 `OpenAISDKStreaming`；如果还需要使用新版 LMDeploy 提供的 PPL 接口，则可以使用 `TurboMindAPIModel`。配置示例如下：

```python
from opencompass.models.turbomind_api import TurboMindAPIModel

models = [
    dict(
        type=TurboMindAPIModel,
        abbr='qwen3.5-35b-a3b-lmdeploy-api',
        model_name='Qwen3.5-35B-A3B',
        api_addr='http://127.0.0.1:23333',
        api_key='sk-admin',
        max_seq_len=262144,
        max_out_len=131072,
        batch_size=8,
        max_workers=8,
        retry=10,
    )
]
```

模型支持范围、服务接口和启动参数可能随推理后端版本变化，请以实际使用版本的官方文档为准。

## 2. 一站式部署与评测

OpenCompass 也提供一站式部署与评测方案：评测任务直接加载本地模型权重，无需预先启动独立服务，适合快速完成单次评测。由于模型部署与评测任务相互耦合，重复加载、资源复用和故障隔离不如独立服务灵活，因此大规模或重复评测更推荐使用独立推理服务。下面给出三种可选配置；运行前应安装相应后端，并确认模型、后端版本与硬件兼容。

### LMDeploy

```python
from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='qwen3.5-35b-a3b-lmdeploy',
        path='Qwen/Qwen3.5-35B-A3B',
        engine_config=dict(session_len=262144, max_batch_size=8, tp=2),
        gen_config=dict(do_sample=False),
        max_seq_len=262144,
        max_out_len=131072,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
    )
]
```

`engine_config.tp` 是 LMDeploy 实际使用的张量并行度，`run_cfg.num_gpus` 是向 OpenCompass Runner 声明的占卡数，两者应保持一致。

### vLLM

```python
from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen3.5-35b-a3b-vllm',
        path='Qwen/Qwen3.5-35B-A3B',
        model_kwargs=dict(
            tensor_parallel_size=2,
            max_model_len=262144,
            trust_remote_code=True,
        ),
        generation_kwargs=dict(temperature=0),
        max_seq_len=262144,
        max_out_len=8192,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
    )
]
```

`tensor_parallel_size` 与 `run_cfg.num_gpus` 同样应保持一致。

### Hugging Face Transformers

不使用推理加速后端时，可以直接通过 Transformers 加载权重：

```python
from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2-0.5b-instruct-hf',
        path='Qwen/Qwen2-0.5B-Instruct',
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

Hugging Face 方式便于兼容性验证，但大模型或大规模评测通常应优先考虑 [LMDeploy](https://github.com/InternLM/lmdeploy) 或 [vLLM](https://github.com/vllm-project/vllm)。运行前请按照相应后端的官方项目和文档完成安装，并确认模型与后端版本兼容。仓库已有一站式部署配置可通过 `python tools/list_configs.py <模型关键字>` 查询。
