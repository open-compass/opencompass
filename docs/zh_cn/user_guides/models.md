# 模型接入总览

模型配置回答三个问题：调用哪一种模型后端、如何生成回复、运行一个模型实例需要多少资源。先根据部署形态选择入口，再调整参数。

| 部署形态                   | 推荐入口                                               | 适用场景                                |
| -------------------------- | ------------------------------------------------------ | --------------------------------------- |
| OpenAI 兼容服务            | `OpenAISDK`、`OpenAISDKResponse`                       | OpenAI 官方接口、自建推理服务、中转网关 |
| 厂商 API                   | `GeminiSDK`、`ClaudeSDK` 等对应模型类                  | 使用厂商原生协议、鉴权和高级能力        |
| 本地推理引擎加速           | LMDeploy 或 vLLM 模型类                                | 大模型、多卡和批量评测                  |
| 本地 Hugging Face 原生加载 | `HuggingFacewithChatTemplate` 或 `HuggingFaceCausalLM` | 小规模验证、兼容性基线                  |
| 仓库尚未支持的后端         | 自定义 `BaseModel` 子类                                | 特殊输入输出或私有运行时                |

接口模型与本地权重的具体配置见下文[接口与推理服务模型](#接口与推理服务模型)与[本地权重模型](#本地权重模型)；自定义后端参阅[新增模型后端](../extension/new_model.md)。

## 所有模型都应关注的字段

下面使用仓库现有的 gpt-6-astra Responses API 配置说明常用字段：

```python
from opencompass.models import OpenAISDKResponse

models = [
    dict(
        type=OpenAISDKResponse,
        abbr='gpt-6-astra-response',
        path='gpt-6-astra',
        key='ENV',  # OPENAI_API_KEY
        openai_api_base='https://api.openai.com/v1',
        max_seq_len=1000000,
        max_out_len=131072,
        query_per_second=1,
        max_workers=8,
        retry=10,
        openai_extra_kwargs=dict(reasoning=dict(effort='max')),
        batch_size=8,
    )
]
```

`type` 选择模型实现，`abbr` 是结果目录和汇总表中的稳定简称，`path` 是服务模型名或本地权重路径。`max_seq_len` 与 `max_out_len` 共同约束输入和生成预算；`batch_size`、`query_per_second`、`max_workers` 和 `retry` 影响吞吐、限流与失败重试。

## 模型配置与数据集配置的边界

模型配置负责模型协议、生成参数、资源和接口行为；数据集配置负责题目内容、Prompt 和评分方式。模型专属特殊 token 应由 tokenizer chat template 或[模型侧对话模板协议](../prompt/meta_template.md)管理，通用题目指令优先放在数据集侧的 [RawPromptTemplate](../prompt/raw_prompt_template.md)。

## 选择现有配置

```bash
python tools/list_configs.py gpt openai
```

打开命中的配置并核对模型版本、依赖、上下文长度和生成参数。目录中存在配置只表示已提供接入样例，不保证它与当前服务版本、显卡或推理框架版本兼容。

## 接口与推理服务模型

API 模型不占用本地 GPU，但受鉴权、限流、网络、服务端模型版本和费用影响。OpenCompass 提供 OpenAI 及 OpenAI SDK 兼容实现，也包含若干厂商适配器；应优先复用最接近目标协议的现有配置。API 模型通常设置 `run_cfg.num_gpus=0`。

### 常用 API 模型类型

#### OpenAISDK

`OpenAISDK` 通过 OpenAI SDK 调用 Chat Completions，是官方 OpenAI、vLLM/LMDeploy serve、中转网关及其他 OpenAI 兼容 endpoint 的通用入口。

```python
from opencompass.models import OpenAISDK

model = dict(type=OpenAISDK, abbr='openai-compatible-model',
             path='served-model-name', key='ENV',
             openai_api_base='https://api.openai.com/v1',
             query_per_second=1, retry=3)
```

重点核对 `key`、`openai_api_base`、`query_per_second`、`retry`、`max_workers` 与 `openai_extra_kwargs`。

#### OpenAISDKStreaming

`OpenAISDKStreaming` 是 `OpenAISDK` 的流式子类，默认设置 `stream=True`，逐 token 接收服务端返回，适合长输出防超时和观察实时生成。

```python
from opencompass.models import OpenAISDKStreaming

model = dict(type=OpenAISDKStreaming, abbr='openai-compatible-streaming',
             path='served-model-name', key='ENV',
             openai_api_base='https://api.openai.com/v1',
             timeout=3600, retry=3)
```

除 `OpenAISDK` 的通用参数外，还可关注 `stream`、`timeout`、`finish_reason_confirm` 和 `stream_chunk_size`。

#### GeminiSDK

`GeminiSDK` 使用 Google Gen AI SDK，密钥从 `GOOGLE_API_KEY` 或 `GEMINI_API_KEY` 读取，并支持 Gemini thinking 配置。

```python
from opencompass.models import GeminiSDK

model = dict(type=GeminiSDK, abbr='gemini-sdk',
             path='gemini-2.5-flash', key='ENV',
             thinking=dict(thinking_budget=1024),
             gemini_extra_kwargs=dict(), retry=3)
```

私有或代理服务可设置 `base_url`；其他生成参数通过 `thinking`、`gemini_extra_kwargs` 和 `client_extra_kwargs` 传入。

#### ClaudeSDK

`ClaudeSDK` 基于 Anthropic Messages SDK，使用 `ANTHROPIC_API_KEY`，支持 thinking 与 Anthropic SDK 的额外请求参数。

```python
from opencompass.models import ClaudeSDK

model = dict(type=ClaudeSDK, abbr='claude-sdk', path='claude-opus-5',
             key='ENV', thinking=dict(type='enabled', budget_tokens=1024),
             claude_extra_kwargs=dict(), retry=3)
```

可通过 `base_url`、`extra_headers` 和 `claude_extra_kwargs` 适配 Anthropic 兼容服务；完整示例见 `configs/models/claude/claude_opus_5.py`。

#### OpenAISDKResponse

`OpenAISDKResponse` 调用 OpenAI Responses API（`client.responses.create`），是本轮基础教程中 gpt-6-astra 的默认模型类。`openai_extra_kwargs` 可直接透传 `reasoning` 等 Responses 参数。

```python
from opencompass.models import OpenAISDKResponse

model = dict(type=OpenAISDKResponse, abbr='gpt-6-astra-response',
             path='gpt-6-astra', key='ENV',
             openai_api_base='https://api.openai.com/v1',
             openai_extra_kwargs=dict(reasoning=dict(effort='max')),
             retry=10)
```

Responses API 的请求关键字放入 `openai_extra_kwargs`；也可使用兼容别名 `response_kwargs`。完整配置见 `configs/models/openai/gpt_6_astra.py`。

### OpenAI 兼容配置

自建或第三方服务只要暴露 OpenAI Chat Completions 兼容接口，通常都可以用 `OpenAISDK` 接入：

```python
from opencompass.models import OpenAISDK

models = [
    dict(
        type=OpenAISDK,
        abbr='my-api-model',
        path='served-model-name',
        key='ENV',
        openai_api_base='https://example.com/v1',
        query_per_second=1,
        max_seq_len=32768,
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=0),
    )
]
```

密钥不要写入提交到 Git 的配置。不同模型类读取的环境变量和 endpoint 字段可能不同，使用前应检查对应类与仓库示例。

### 通过部署推理加速服务来运行

当显存或吞吐不足、需要多人共享模型服务，或希望让评测与推理解耦时，可以先用 vLLM 或 LMDeploy 启动 OpenAI 兼容服务，再让 OpenCompass 通过 `OpenAISDK` 请求服务。

下面以 LMDeploy 为例启动服务：

```bash
lmdeploy serve api_server Qwen/Qwen3.5-35B-A3B \
    --model-name Qwen3.5-35B-A3B \
    --tp 2 \
    --server-port 23333
```

`--tp` 设置张量并行，`--session-len` 设置最大上下文窗口，`--cache-max-entry-count` 调整 k/v cache 的内存使用比例。服务启动后，将模型配置指向该地址：

```python
from opencompass.models import OpenAISDK

models = [
    dict(
        type=OpenAISDK,
        abbr='Qwen3.5-35B-A3B-LMDeploy-API',
        path='Qwen3.5-35B-A3B',
        key='EMPTY',
        openai_api_base='http://0.0.0.0:23333/v1',
        tokenizer_path='Qwen/Qwen3.5-35B-A3B',
        query_per_second=1,
        max_out_len=8192,
        max_seq_len=131072,
        temperature=0.6,
        batch_size=8,
        retry=3,
        run_cfg=dict(num_gpus=0),
    )
]
```

新数据集使用 RawPromptTemplate 直接产生 `role/content` 消息时通常不需要额外的 `api_meta_template`；传统 PromptTemplate 需要角色映射时，再配置模型侧对话模板协议。若希望部署与评测在同一次 `opencompass` 运行中完成，参阅[本地模型的一站式部署和评测](../faq/local_model_one_stop.md)。

### 并发、重试与确定性

`batch_size`、`query_per_second`、`max_workers` 和 Runner worker 数会共同影响请求并发。出现 429、超时或服务端错误时，应先按服务配额降低并发，再调整重试；无限重试可能造成重复费用。

即使 `temperature=0`，服务端升级、路由、推理实现和安全策略也可能改变回复。正式结果应记录服务模型的精确名称或版本、API base、供应方和区域、请求时间、生成参数、重试策略、配置快照及异常请求数量。

### 输入协议

新数据集优先使用 [RawPromptTemplate](../prompt/raw_prompt_template.md) 产生 `role/content` 消息。若模型配置还会通过 `meta_template` 追加 system/user 消息，必须预览最终输入，避免重复指令。推理前用极小数据集验证鉴权、消息格式、停止条件和错误处理，再扩大并发。

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

CLI 的 `--accelerator vllm` 或 `--accelerator lmdeploy` 可以在一次 OpenCompass 运行中转换部分 Hugging Face 模型配置。转换能力、限制及完整示例见[本地模型的一站式部署和评测](../faq/local_model_one_stop.md)。
