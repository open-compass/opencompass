# Model Configuration Overview

A standard OpenCompass model configuration specifies the model class, model hyperparameters such as context length, inference concurrency, deployment resources, and related settings. The principal model categories are listed below.

| Deployment form                   | Recommended entry point                                                                | Typical use case                                                      |
| --------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| OpenAI-compatible service         | `OpenAISDK`, `OpenAISDKStreaming`, `OpenAISDKResponse`                                 | OpenAI APIs, self-hosted OpenAI-compatible services, and API gateways |
| Other vendor APIs                 | Model classes such as `GeminiSDK` and `ClaudeSDK`                                      | Vendor-native protocols, authentication, and parameters               |
| Local accelerated inference       | `TurboMindModelwithChatTemplate` (LMDeploy), `VLLMwithChatTemplate` (vLLM), and others | One-stop deployment and evaluation                                    |
| Native local Hugging Face loading | `HuggingFacewithChatTemplate` or `HuggingFaceCausalLM`                                 | One-stop deployment and evaluation                                    |

The following sections describe API-based model invocation and one-stop local-weight loading within the OpenCompass process. For custom backends, see [Adding a Model Backend](../extension/new_model.md).

## 1. API Models

For an API model, a remote service performs inference while OpenCompass orchestrates requests and stores results. Before configuring the model, verify the service URL, model name, credentials, request limits, and timeout settings. API models normally consume no GPU on the evaluation host, so `run_cfg.num_gpus` can be set to `0`.

### Common Parameters

Common fields are listed below. Supported parameters vary by model class; consult the corresponding constructor and existing repository configurations.

| Parameter          | Description                                                                                                                                                                            |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`             | Model class used by OpenCompass, such as `OpenAISDKResponse`.                                                                                                                          |
| `abbr`             | Short model name used in output directories, result files, and summary tables.                                                                                                         |
| `path`             | Model name exposed by the service. Some classes use another field; for example, `TurboMindAPIModel` uses `model_name`.                                                                 |
| `key`              | API key. Pass a string directly or use `os.getenv()` to read a custom environment variable.                                                                                            |
| `max_seq_len`      | Maximum sequence length accepted by the model. The combined input and output should not exceed this value.                                                                             |
| `max_out_len`      | Maximum number of tokens generated per request. An explicit value in the dataset-side Inferencer takes precedence.                                                                     |
| `temperature`      | Sampling temperature. The valid range and exact semantics depend on the service implementation.                                                                                        |
| `query_per_second` | Maximum request rate per second. Excessive values may trigger service-side rate limiting.                                                                                              |
| `batch_size`       | Batch size used by the `Inferencer`. With `OpenICLInferTask`, it is equivalent to the number of concurrent API threads.                                                                |
| `max_workers`      | Maximum number of concurrent API request threads, for model classes that declare this parameter. With `OpenICLInferConcurrentTask`, `max_workers` is the sole concurrency control for that model. |
| `retry`            | Maximum number of retries after a request failure.                                                                                                                                     |
| `tokenizer_path`   | Tokenizer name or path used to estimate input length. It is normally required when using a gateway or a custom served-model name.                                                      |

### OpenAISDK: Chat Completions

`OpenAISDK` calls the Chat Completions endpoint through `client.chat.completions.create`. It supports both OpenAI services and compatible services implementing the same interface:

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

The principal parameters are:

- `openai_extra_kwargs`: fields in this dictionary are inserted directly into the Chat Completions request body.
- `extra_body`: additional request-body fields exposed by a compatible service.
- `status_code_mappings`: maps selected HTTP error status codes to fixed model outputs. In the example above, a 400 response immediately returns the mapped text instead of retrying; unmapped errors continue to follow the `retry` setting. This is suitable for placeholder outputs for predictable service refusals such as content filtering. Do not map authentication failures, rate limits, or service faults to normal outputs.

### OpenAISDKStreaming: Streaming Chat Completions

`OpenAISDKStreaming` inherits from `OpenAISDK`, receives a streaming Chat Completions response, and passes the complete text to the evaluation workflow after the stream ends. It is suitable when the service requires streaming or generation may take a long time:

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

In addition to `OpenAISDK` parameters, the streaming model provides:

- `finish_reason_confirm`: whether the final stream response must contain `finish_reason`; defaults to `True`. If the stream ends without this field, the response is considered incomplete and retried. When set to `False`, the text collected so far is returned.
- `verbose`: when `True`, logs request start, finish reason, and elapsed time, and also prints reasoning and answer text from each streaming response chunk to the terminal in real time. By contrast, `OpenAISDK.verbose` logs only request and response processing and does not print generated content chunk by chunk. This option is useful for debugging long generations; terminal output from concurrent requests may be interleaved.

### OpenAISDKResponse: Responses API

`OpenAISDKResponse` calls the OpenAI Responses API through `client.responses.create`. Pass Responses-specific parameters through `openai_extra_kwargs`; see the [OpenAI Responses API reference](https://developers.openai.com/api/reference/resources/responses/methods/create/) for available fields. Example:

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

### Other Vendor APIs

Authentication, model naming, and generation parameters vary substantially across vendor-native SDKs. OpenCompass converts the unified evaluation input into the corresponding protocol, while the vendor's official API documentation remains authoritative. Before use, verify both the installed SDK version and the permissions of the API account. The following examples cover two common vendor SDKs.

Gemini SDK example:

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

Claude SDK example:

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

### Evaluating a Locally Deployed Inference Service

If local GPUs are available and you want to evaluate a locally deployed model, decouple model serving from evaluation: start an independent service with an inference backend such as LMDeploy, then call it through an OpenCompass API model class. The following example deploys a model with LMDeploy:

```bash
lmdeploy serve api_server Qwen/Qwen3.5-35B-A3B \
    --model-name Qwen3.5-35B-A3B \
    --tp 2 \
    --server-port 23333
```

After confirming that the service is reachable, select a model class according to the service interface and evaluation method. Use `OpenAISDK` or `OpenAISDKStreaming` for an OpenAI-compatible generation interface. If the newer LMDeploy PPL endpoint is also required, use `TurboMindAPIModel`. Example:

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

Supported models, service interfaces, and launch parameters may vary by inference-backend version. Consult the documentation for the version in use.

## 2. One-Stop Deployment and Evaluation

OpenCompass also supports one-stop deployment and evaluation: an evaluation task loads local model weights directly without requiring a separately launched service, which is convenient for a single evaluation. Because serving and evaluation are coupled, repeated loading, resource reuse, and fault isolation are less flexible than with an independent service. An independent inference service is therefore preferred for large-scale or repeated evaluations. Choose one of the following configurations, install the corresponding backend, and verify compatibility among the model, backend version, and hardware.

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

`engine_config.tp` is the tensor-parallel degree actually used by LMDeploy, while `run_cfg.num_gpus` declares the number of GPUs reserved by the OpenCompass Runner. Keep the two values consistent.

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

Keep `tensor_parallel_size` consistent with `run_cfg.num_gpus` as well.

### Hugging Face Transformers

Without an accelerated inference backend, load weights directly through Transformers:

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

The Hugging Face path is useful for compatibility validation, but large models and large-scale evaluations should normally use [LMDeploy](https://github.com/InternLM/lmdeploy) or [vLLM](https://github.com/vllm-project/vllm). Follow the corresponding project's installation instructions and verify model/backend compatibility. Query available one-stop deployment configurations with `python tools/list_configs.py <model-keyword>`.
