# One-Stop Deployment and Evaluation of Local Models

This page describes loading, running, and evaluating a local model within one `opencompass` invocation. OpenCompass can convert a Hugging Face model configuration to a vLLM or LMDeploy backend; the evaluation process manages the model directly, so no independent inference service needs to be started beforehand.

If you want to start a shared service first and call it from the evaluation process through an API, use the [service deployment approach in Model Integration](../user_guides/models.md#running-through-a-deployed-accelerated-inference-service).

## Preparing an Inference Backend

[LMDeploy](https://github.com/InternLM/lmdeploy) and [vLLM](https://github.com/vllm-project/vllm) both load local weights with a high-throughput inference engine. First confirm that the target model is supported by the framework, then prepare the environment according to its installation documentation. You can also install directly:

```bash
pip install lmdeploy
pip install vllm
```

## Switching Backends from the Command Line

The following example uses Qwen3.5-35B-A3B and the GSM8K demo dataset. Create `eval_gsm8k.py`:

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

By default, this uses the Hugging Face Transformers backend:

```bash
opencompass eval_gsm8k.py
```

Add `-a` or `--accelerator` to switch the backend for this run:

```bash
opencompass eval_gsm8k.py -a vllm
opencompass eval_gsm8k.py -a lmdeploy
```

During conversion, `HuggingFacewithChatTemplate` becomes `VLLMwithChatTemplate` or `TurboMindModelwithChatTemplate`:

- `tensor_parallel_size` / `tp` comes from `run_cfg.num_gpus`.
- `max_model_len` / `session_len` comes from `max_seq_len`.
- Generation arguments are preserved where possible.
- LMDeploy uses greedy decoding by default; set `do_sample=True` when sampling is required.

CLI conversion covers only some model types and arguments. The repository provides `configs/models/qwen3/vllm_qwen3_5_35b_a3b.py` and `configs/models/qwen3/lmdeploy_qwen3_5_35b_a3b.py`, which contain complete settings such as `enable_thinking` and chain-of-thought postprocessing. For formal evaluation, inspect and directly import the corresponding backend configuration where possible.

## Performance Reference

The following is a historical reference result for Llama-3-8B-Instruct on one A800 GPU with GSM8K:

| Inference backend | Accuracy | Inference time (min:sec) | Speedup over Hugging Face |
| ----------------- | -------: | -----------------------: | ------------------------: |
| Hugging Face      |    74.22 |                    24:26 |                      1.0× |
| LMDeploy          |    73.69 |                    11:15 |                      2.2× |
| vLLM              |    72.63 |                    07:52 |                      3.1× |

Actual speedup depends on model architecture, GPU, dataset, context length, and sampling configuration. Differences between backend sampling implementations can also affect accuracy.

To run the model as an independent service shared by multiple users, see [Running Through a Deployed Accelerated Inference Service](../user_guides/models.md#running-through-a-deployed-accelerated-inference-service).
