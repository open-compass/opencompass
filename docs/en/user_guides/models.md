# Model Integration Overview

A model configuration answers three questions: which model backend to invoke, how replies are generated, and how many resources one model instance requires. Choose an entry point by deployment form, then tune its arguments.

| Deployment form                    | Recommended entry point                                | Use case                                                          |
| ---------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------- |
| OpenAI-compatible service          | `OpenAISDK`, `OpenAISDKResponse`                       | Official OpenAI API, self-hosted inference service, or gateway    |
| Vendor API                         | Corresponding class such as `GeminiSDK` or `ClaudeSDK` | Native vendor protocol, authentication, and advanced capabilities |
| Accelerated local inference engine | LMDeploy or vLLM model class                           | Large models, multiple GPUs, and batch evaluation                 |
| Native local Hugging Face loading  | `HuggingFacewithChatTemplate` or `HuggingFaceCausalLM` | Small-scale validation and compatibility baseline                 |
| Unsupported backend                | Custom `BaseModel` subclass                            | Specialized input/output or private runtime                       |

See [API and Inference-Service Models](#api-and-inference-service-models) and [Local-Weight Models](#local-weight-models) below for concrete configuration, and [Adding a Model Backend](../extension/new_model.md) for a custom backend.

## Fields Every Model Should Consider

The following repository configuration for gpt-6-astra through the Responses API illustrates common fields:

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

`type` selects the implementation, `abbr` is the stable short name used in result directories and summary tables, and `path` is a service model name or local-weight path. `max_seq_len` and `max_out_len` jointly constrain input and generation budgets; `batch_size`, `query_per_second`, `max_workers`, and `retry` affect throughput, rate limiting, and failure retries.

## Boundary Between Model and Dataset Configurations

A model configuration owns the model protocol, generation arguments, resources, and API behavior. A dataset configuration owns question content, prompts, and scoring. Model-specific special tokens should be managed by the tokenizer chat template or [Model-Side Conversation Template Protocol](../prompt/meta_template.md); general question instructions should normally live in the Dataset's [RawPromptTemplate](../prompt/raw_prompt_template.md).

## Selecting an Existing Configuration

```bash
python tools/list_configs.py gpt openai
```

Open each match and verify model version, dependencies, context length, and generation arguments. A configuration in the directory is an integration example, not a guarantee of compatibility with the current service version, GPU, or inference-framework version.

## API and Inference-Service Models

API models consume no local GPU, but are affected by authentication, rate limits, networking, server-side model versions, and cost. OpenCompass provides OpenAI and OpenAI-SDK-compatible implementations plus several vendor adapters. Reuse the existing configuration closest to the target protocol. API models normally set `run_cfg.num_gpus=0`.

### Common API Model Types

#### OpenAISDK

`OpenAISDK` calls Chat Completions through the OpenAI SDK and is the general entry point for official OpenAI, vLLM/LMDeploy serve, gateways, and other OpenAI-compatible endpoints.

```python
from opencompass.models import OpenAISDK

model = dict(type=OpenAISDK, abbr='openai-compatible-model',
             path='served-model-name', key='ENV',
             openai_api_base='https://api.openai.com/v1',
             query_per_second=1, retry=3)
```

Pay particular attention to `key`, `openai_api_base`, `query_per_second`, `retry`, `max_workers`, and `openai_extra_kwargs`.

#### OpenAISDKStreaming

`OpenAISDKStreaming` is a streaming subclass of `OpenAISDK`. It sets `stream=True` by default and receives server output token by token, which is useful for avoiding long-output timeouts and observing generation in real time.

```python
from opencompass.models import OpenAISDKStreaming

model = dict(type=OpenAISDKStreaming, abbr='openai-compatible-streaming',
             path='served-model-name', key='ENV',
             openai_api_base='https://api.openai.com/v1',
             timeout=3600, retry=3)
```

In addition to common `OpenAISDK` arguments, consider `stream`, `timeout`, `finish_reason_confirm`, and `stream_chunk_size`.

#### GeminiSDK

`GeminiSDK` uses the Google Gen AI SDK, reads a key from `GOOGLE_API_KEY` or `GEMINI_API_KEY`, and supports Gemini thinking configuration.

```python
from opencompass.models import GeminiSDK

model = dict(type=GeminiSDK, abbr='gemini-sdk',
             path='gemini-2.5-flash', key='ENV',
             thinking=dict(thinking_budget=1024),
             gemini_extra_kwargs=dict(), retry=3)
```

Private or proxied services can set `base_url`; pass other generation arguments through `thinking`, `gemini_extra_kwargs`, and `client_extra_kwargs`.

#### ClaudeSDK

`ClaudeSDK` is based on the Anthropic Messages SDK, uses `ANTHROPIC_API_KEY`, and supports thinking and extra Anthropic SDK request arguments.

```python
from opencompass.models import ClaudeSDK

model = dict(type=ClaudeSDK, abbr='claude-sdk', path='claude-opus-5',
             key='ENV', thinking=dict(type='enabled', budget_tokens=1024),
             claude_extra_kwargs=dict(), retry=3)
```

Use `base_url`, `extra_headers`, and `claude_extra_kwargs` for an Anthropic-compatible service. See `configs/models/claude/claude_opus_5.py` for a complete example.

#### OpenAISDKResponse

`OpenAISDKResponse` calls the OpenAI Responses API (`client.responses.create`) and is the default model class for gpt-6-astra in the basic tutorial. `openai_extra_kwargs` forwards Responses arguments such as `reasoning` directly.

```python
from opencompass.models import OpenAISDKResponse

model = dict(type=OpenAISDKResponse, abbr='gpt-6-astra-response',
             path='gpt-6-astra', key='ENV',
             openai_api_base='https://api.openai.com/v1',
             openai_extra_kwargs=dict(reasoning=dict(effort='max')),
             retry=10)
```

Put Responses API request keywords in `openai_extra_kwargs`; the compatible alias `response_kwargs` is also supported. See `configs/models/openai/gpt_6_astra.py` for the complete configuration.

### OpenAI-Compatible Configuration

A self-hosted or third-party service exposing an OpenAI Chat Completions-compatible endpoint can normally be integrated through `OpenAISDK`:

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

Do not commit keys to Git. Different model classes may read different environment variables and endpoint fields; inspect the class and repository example before use.

### Running Through a Deployed Accelerated Inference Service

When memory or throughput is insufficient, a model service must be shared, or evaluation and inference should be decoupled, first launch an OpenAI-compatible service with vLLM or LMDeploy, then have OpenCompass request it through `OpenAISDK`.

The following example launches an LMDeploy service:

```bash
lmdeploy serve api_server Qwen/Qwen3.5-35B-A3B \
    --model-name Qwen3.5-35B-A3B \
    --tp 2 \
    --server-port 23333
```

`--tp` sets tensor parallelism, `--session-len` sets the maximum context window, and `--cache-max-entry-count` controls the fraction of memory used by the k/v cache. Point the model configuration to the service after it starts:

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

A new dataset using RawPromptTemplate to produce `role/content` messages normally needs no extra `api_meta_template`. Configure the Model-Side Conversation Template Protocol only when a traditional PromptTemplate requires role mapping. To deploy and evaluate within one `opencompass` invocation, see [One-Stop Deployment and Evaluation of Local Models](../faq/local_model_one_stop.md).

### Concurrency, Retries, and Determinism

`batch_size`, `query_per_second`, `max_workers`, and the Runner worker count jointly affect request concurrency. For 429 responses, timeouts, or server errors, first reduce concurrency according to the service quota, then tune retries. Unlimited retries can create duplicate charges.

Even with `temperature=0`, server upgrades, routing, inference implementations, and safety policy can change replies. Formal results should record the exact service model name or version, API base, provider and region, request time, generation arguments, retry strategy, configuration snapshot, and number of anomalous requests.

### Input Protocol

New datasets should produce `role/content` messages with [RawPromptTemplate](../prompt/raw_prompt_template.md). If the model configuration also adds system/user messages through `meta_template`, preview the final input to avoid duplicated instructions. Before increasing concurrency, validate authentication, message format, stopping conditions, and error handling with a very small dataset.

## Local-Weight Models

### Transformers Chat Models

For an instruction model that supports the Hugging Face chat template, prefer `HuggingFacewithChatTemplate`:

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

`path` can be a Hub ID or a local weights directory. Enable `trust_remote_code=True` only when repository-defined model code is required, and pin a trusted revision.

### Base Models and PPL Evaluation

Use `HuggingFaceCausalLM` for a base model or a discriminative task requiring token probabilities. Chat models normally use generative configurations only; do not use a PPL configuration for a model that exposes only a chat endpoint and cannot return token probabilities.

### Length, Batching, and Memory

- `max_seq_len` should cover both input and generation budgets; truncation changes the evaluated question.
- A small `max_out_len` truncates answers, while an excessive value reduces throughput and increases cost.
- `batch_size` is the batch size OpenCompass passes to the model and does not guarantee identical backend batching.
- Left padding is normally suitable for decoder-only batch generation; actual padding support depends on the model implementation.
- `run_cfg.num_gpus` tells the Runner how many GPUs one task occupies; it does not automatically enable tensor parallelism in an arbitrary implementation.

First run `--dry-run` on a demo dataset, then inspect prompts, truncation, stop words, and generation results on a small sample. For insufficient memory, reduce batch size and context budget before considering quantization or a high-throughput backend.

### Multiple GPUs and Parallelism

When one model instance is split across GPUs with tensor parallelism, configure parallelism inside the model and declare its GPU occupancy through `run_cfg`; verify that the two agree:

```python
models = [
    dict(
        type=...,                                   # e.g. VLLMwithChatTemplate
        abbr='my-model-tp2',
        path='org/model',
        model_kwargs=dict(tensor_parallel_size=2),  # Backend tensor parallelism
        run_cfg=dict(num_gpus=2),                   # GPU count declared to scheduler
    )
]
```

CLI `--accelerator vllm` or `--accelerator lmdeploy` can convert some Hugging Face model configurations within one OpenCompass run. See [One-Stop Deployment and Evaluation of Local Models](../faq/local_model_one_stop.md) for conversion support, limitations, and a complete example.
