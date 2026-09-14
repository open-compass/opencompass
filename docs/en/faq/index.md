# Frequently Asked Questions

## General

### What is the difference and relationship between `ppl` and `gen`?

`ppl` abbreviates perplexity, a metric for language-modeling ability. In OpenCompass it normally describes a multiple-choice method: given a context and n options, concatenate each option to the context to form n sequences, compute their perplexities, and select the option belonging to the lowest-perplexity sequence. Its postprocessing is direct and highly deterministic.

`gen` abbreviates generate. Given a context, the model's continuation is treated as its answer. The generated string normally requires more substantial postprocessing before an answer can be extracted reliably for evaluation.

Single-answer multiple choice and some choice-like tasks use `ppl` for base models; multiple-answer and non-choice tasks use `gen`. All tasks use `gen` for chat models because many commercial API models do not expose a `ppl` interface. Exceptions exist—for example, `gen` is also used when a base model should output reasoning such as “Let's think step by step.” The general rule is:

|            | ppl      | gen                  |
| ---------- | -------- | -------------------- |
| Base model | MCQ only | Tasks other than MCQ |
| Chat model | None     | All tasks            |

Conditional log probability, `clp`, is closely related to `ppl`: given a context, it computes the probability of the next token. It also applies only to multiple choice and scores only tokens corresponding to option labels, choosing the highest-probability label. `clp` needs only one inference rather than n, but is sensitive to tokenizer behavior; spaces around a label can change its encoding and make results unreliable. OpenCompass therefore uses `clp` rarely.

### How does OpenCompass control the shot count of few-shot evaluation?

A dataset configuration has a `retriever` field describing how dataset samples are selected as context examples. The common `FixKRetriever` uses fixed k samples and therefore performs k-shot evaluation. `ZeroRetriever` uses no retrieved sample and usually means 0-shot.

However, in-context examples can also be written directly in the dataset template. Such a configuration also uses `ZeroRetriever`, but is not necessarily 0-shot; determine it from the actual template. See [Few-Shot Example Insertion in Prompt Templates](../prompt/raw_prompt_template.md#inserting-few-shot-examples-ice).

### What is the default OpenCompass task-partitioning logic?

OpenCompass uses `NumWorkerPartitioner` by default. Evaluation combines a series of models with a series of datasets and runs every model on every dataset. For one model, OpenCompass divides work among `--max-num-workers` tasks, or `infer.runner.max_num_workers` in configuration. To balance runtime, every task receives portions of the datasets:

![num_worker_partitioner](https://github.com/open-compass/opencompass/assets/17680578/68c57a57-0804-4865-a0c6-133e1657b9fc)

### Why are some inference log files absent when OpenCompass runs through Slurm or a similar backend?

A log filename represents a task rather than one dataset shard. A Partitioner may combine several small jobs into one larger task, whose name is normally derived from the first dataset. Later datasets in that task therefore have no same-named log; their output is written into the first dataset's log.

### How does resume work in OpenCompass?

Using `--reuse` / `-r` enables resume. OpenCompass first configures models and datasets from the latest configuration, then the Partitioner determines shard size and creates shards. Each shard is inspected in order: a completed shard is skipped, while a missing or incomplete shard enters the pending list. An incomplete task normally has an output filename beginning with `tmp_`; the model continues after the largest completed record index until the shard finishes.

Consequently:

- When resuming from an existing output directory, do not change the Partitioner sharding method or `--max-num-workers`, unless `tools/prediction_merger.py` has been used.
- If the dataset changes, do not resume. Rerun everything, deleting old outputs selectively only when appropriate.

### How does OpenCompass allocate GPUs?

OpenCompass processes evaluation requests in units called tasks. Each task is an independent model-dataset combination. Its GPU resource requirement is determined by the evaluated model's `num_gpus` argument.

During evaluation, multiple workers run tasks concurrently and acquire GPU resources when available. OpenCompass therefore attempts to make full use of visible GPUs.

For example, on a local machine with 8 GPUs, if every task requests 4 GPUs, OpenCompass runs 2 tasks concurrently by default and uses all 8 GPUs. Setting `--max-num-workers` to 1 allows only one task and uses 4 GPUs at a time.

### How can I control the number of GPUs occupied by OpenCompass?

There is currently no direct argument specifying the total number of GPUs available to OpenCompass, but it can be controlled indirectly.

**For local evaluation:** set `CUDA_VISIBLE_DEVICES` to limit GPU visibility. For example, `CUDA_VISIBLE_DEVICES=0,1,2,3 opencompass ...` exposes only four GPUs, so OpenCompass cannot use more than those four simultaneously.

**For Slurm or DLC:** OpenCompass does not directly own the resource pool, but `--max-num-workers` limits concurrently submitted evaluation tasks. If each task needs 4 GPUs and the desired total is 8, set `--max-num-workers` to 2.

### `libGL.so.1` cannot be found

`opencv-python` depends on dynamic libraries that may be absent. The simplest solution is to replace it with `opencv-python-headless`:

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

Alternatively, install the libraries reported by the error:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

### Error: mkl-service + Intel(R) MKL

The complete error is:

```text
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp-a34b3233.so.1 library.
	Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
```

Set `MKL_SERVICE_FORCE_INTEL=1` to resolve it.

## Network

### `ConnectionResetError(104, 'Connection reset by peer')` or a Hugging Face `MaxRetryError`

Because of Hugging Face behavior, OpenCompass needs network access when some datasets and models are first loaded, and connects to Hugging Face during startup. Options are:

- Configure a proxy with `http_proxy` and `https_proxy`.

- Reuse cache files from another machine. Run the experiment on a machine with Hugging Face access, then copy or symlink its cache, normally `~/.cache/huggingface/` ([documentation](https://huggingface.co/docs/datasets/cache#cache-directory)), to the offline machine. With a complete cache, start in offline mode:

  ```bash
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 HF_HUB_OFFLINE=1 opencompass ...
  ```

  A missing model or dataset file in the cache still raises an error.

- Use a mirror available in mainland China, such as [hf-mirror](https://hf-mirror.com/):

  ```bash
  HF_ENDPOINT=https://hf-mirror.com opencompass ...
  ```

### My server cannot access the Internet. How can I use OpenCompass?

As described in the preceding network question, prepare cache files on another machine and copy them to this server.

## Efficiency

### Why does OpenCompass split an evaluation request into tasks?

A comprehensive linear evaluation of an LLM can take a long time because both evaluation duration and dataset count are large. OpenCompass divides a request into independent tasks and dispatches them across GPU groups or nodes, enabling full parallelism and maximizing resource efficiency.

### How does task partitioning work?

Each task represents a specific combination of a model and a portion of a dataset waiting for evaluation. OpenCompass provides partitioning strategies for different scenarios. During inference, the main strategies balance task size or computational cost, estimated heuristically from dataset size and inference type.

### Why does evaluating an LLM take so long in OpenCompass?

Check:

1. Whether a high-throughput inference backend such as vLLM or LMDeploy is being used.
2. When using native Hugging Face execution, whether `batch_size=1` is unnecessarily limiting throughput; increase it where memory permits.
3. Whether substantial time is spent on networking or model downloads from Hugging Face.
4. Whether model output is unexpectedly long, especially when a base model continues generating and answering additional questions. Add `stopping_criteria` to the dataset configuration where appropriate.

If none of these checks resolves the issue, consider filing a bug report.

## Models

### How do I use a locally downloaded Hugging Face model?

Specify the local model path explicitly:

```bash
opencompass --datasets siqa_gen winograd_ppl --hf-type base --hf-path /path/to/model
```

## Datasets

### How do I build my own evaluation dataset?

- Objective datasets: [Adding a Dataset](../extension/new_dataset.md)
- Subjective datasets: [Subjective Evaluation Guidance](../evaluation/subjective_evaluation.md)
