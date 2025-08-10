# FAQ

## General

### What are the differences and connections between `ppl` and `gen`?

`ppl` stands for perplexity, an index used to evaluate a model's language modeling capabilities. In the context of OpenCompass, it generally refers to a method of answering multiple-choice questions: given a context, the model needs to choose the most appropriate option from multiple choices. In this case, we concatenate the n options with the context to form n sequences, then calculate the model's perplexity for these n sequences. We consider the option corresponding to the sequence with the lowest perplexity as the model's reasoning result for this question. This evaluation method is simple and direct in post-processing, with high certainty.

`gen` is an abbreviation for generate. In the context of OpenCompass, it refers to the model's continuation writing result given a context as the reasoning result for a question. Generally, the string obtained from continuation writing requires a heavier post-processing process to extract reliable answers and complete the evaluation.

In terms of usage, multiple-choice questions and some multiple-choice-like questions of the base model use `ppl`, while the base model's multiple-selection and non-multiple-choice questions use `gen`. All questions of the chat model use `gen`, as many commercial API models do not expose the `ppl` interface. However, there are exceptions, such as when we want the base model to output the problem-solving process (e.g., Let's think step by step), we will also use `gen`, but the overall usage is as shown in the following table:

|            | ppl            | gen                  |
| ---------- | -------------- | -------------------- |
| Base Model | Only MCQ Tasks | Tasks Other Than MCQ |
| Chat Model | None           | All Tasks            |

Similar to `ppl`, conditional log probability (`clp`) calculates the probability of the next token given a context. It is also only applicable to multiple-choice questions, and the range of probability calculation is limited to the tokens corresponding to the option numbers. The option corresponding to the token with the highest probability is considered the model's reasoning result. Compared to `ppl`, `clp` calculation is more efficient, requiring only one inference, whereas `ppl` requires n inferences. However, the drawback is that `clp` is subject to the tokenizer. For example, the presence or absence of space symbols before and after an option can change the tokenizer's encoding result, leading to unreliable test results. Therefore, `clp` is rarely used in OpenCompass.

### How does OpenCompass control the number of shots in few-shot evaluations?

In the dataset configuration file, there is a retriever field indicating how to recall samples from the dataset as context examples. The most commonly used is `FixKRetriever`, which means using a fixed k samples, hence k-shot. There is also `ZeroRetriever`, which means not using any samples, which in most cases implies 0-shot.

On the other hand, in-context samples can also be directly specified in the dataset template. In this case, `ZeroRetriever` is also used, but the evaluation is not 0-shot and needs to be determined based on the specific template. Refer to [prompt](../prompt/prompt_template.md) for more details

### How does OpenCompass allocate GPUs?

OpenCompass processes evaluation requests using the unit termed as "task". Each task is an independent combination of model(s) and dataset(s). The GPU resources needed for a task are determined entirely by the model being evaluated, specifically by the `num_gpus` parameter.

During evaluation, OpenCompass deploys multiple workers to execute tasks in parallel. These workers continuously try to secure GPU resources and run tasks until they succeed. As a result, OpenCompass always strives to leverage all available GPU resources to their maximum capacity.

For instance, if you're using OpenCompass on a local machine equipped with 8 GPUs, and each task demands 4 GPUs, then by default, OpenCompass will employ all 8 GPUs to concurrently run 2 tasks. However, if you adjust the `--max-num-workers` setting to 1, then only one task will be processed at a time, utilizing just 4 GPUs.

### Why doesn't the GPU behavior of HuggingFace models align with my expectations?

This is a complex issue that needs to be explained from both the supply and demand sides:

The supply side refers to how many tasks are being run. A task is a combination of a model and a dataset, and it primarily depends on how many models and datasets need to be tested. Additionally, since OpenCompass splits a larger task into multiple smaller tasks, the number of data entries per sub-task (`--max-partition-size`) also affects the number of tasks. (The `--max-partition-size` is proportional to the actual number of data entries, but the relationship is not 1:1).

The demand side refers to how many workers are running. Since OpenCompass instantiates multiple models for inference simultaneously, we use `--hf-num-gpus` to specify how many GPUs each instance uses. Note that `--hf-num-gpus` is a parameter specific to HuggingFace models and setting this parameter for non-HuggingFace models will not have any effect. We also use `--max-num-workers` to indicate the maximum number of instances running at the same time. Lastly, due to issues like GPU memory and insufficient load, OpenCompass also supports running multiple instances on the same GPU, which is managed by the parameter `--max-num-workers-per-gpu`. Therefore, it can be generally assumed that we will use a total of `--hf-num-gpus` * `--max-num-workers` / `--max-num-workers-per-gpu` GPUs.

In summary, when tasks run slowly or the GPU load is low, we first need to check if the supply is sufficient. If not, consider reducing `--max-partition-size` to split the tasks into finer parts. Next, we need to check if the demand is sufficient. If not, consider increasing `--max-num-workers` and `--max-num-workers-per-gpu`. Generally, **we set `--hf-num-gpus` to the minimum value that meets the demand and do not adjust it further.**

### How do I control the number of GPUs that OpenCompass occupies?

Currently, there isn't a direct method to specify the number of GPUs OpenCompass can utilize. However, the following are some indirect strategies:

**If evaluating locally:**
You can limit OpenCompass's GPU access by setting the `CUDA_VISIBLE_DEVICES` environment variable. For instance, using `CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py ...` will only expose the first four GPUs to OpenCompass, ensuring it uses no more than these four GPUs simultaneously.

**If using Slurm or DLC:**
Although OpenCompass doesn't have direct access to the resource pool, you can adjust the `--max-num-workers` parameter to restrict the number of evaluation tasks being submitted simultaneously. This will indirectly manage the number of GPUs that OpenCompass employs. For instance, if each task requires 4 GPUs, and you wish to allocate a total of 8 GPUs, then you should set `--max-num-workers` to 2.

### `libGL.so.1` not foune

opencv-python depends on some dynamic libraries that are not present in the environment. The simplest solution is to uninstall opencv-python and then install opencv-python-headless.

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

Alternatively, you can install the corresponding dependency libraries according to the error message

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

## Network

### My tasks failed with error: `('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))` or `urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='cdn-lfs.huggingface.co', port=443)`

Because of HuggingFace's implementation, OpenCompass requires network (especially the connection to HuggingFace) for the first time it loads some datasets and models. Additionally, it connects to HuggingFace each time it is launched. For a successful run, you may:

- Work behind a proxy by specifying the environment variables `http_proxy` and `https_proxy`;
- Use the cache files from other machines. You may first run the experiment on a machine that has access to the Internet, and then copy the cached files to the offline one. The cached files are located at `~/.cache/huggingface/` by default ([doc](https://huggingface.co/docs/datasets/cache#cache-directory)). When the cached files are ready, you can start the evaluation in offline mode:
  ```python
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run.py ...
  ```
  With which no more network connection is needed for the evaluation. However, error will still be raised if the files any dataset or model is missing from the cache.
- Use mirror like [hf-mirror](https://hf-mirror.com/)
  ```python
  HF_ENDPOINT=https://hf-mirror.com python run.py ...
  ```

### My server cannot connect to the Internet, how can I use OpenCompass?

Use the cache files from other machines, as suggested in the answer to [Network-Q1](#my-tasks-failed-with-error-connection-aborted-connectionreseterror104-connection-reset-by-peer-or-urllib3exceptionsmaxretryerror-httpsconnectionpoolhostcdn-lfshuggingfaceco-port443).

### In evaluation phase, I'm running into an error saying that `FileNotFoundError: Couldn't find a module script at opencompass/accuracy.py. Module 'accuracy' doesn't exist on the Hugging Face Hub either.`

HuggingFace tries to load the metric (e.g. `accuracy`) as an module online, and it could fail if the network is unreachable. Please refer to [Network-Q1](#my-tasks-failed-with-error-connection-aborted-connectionreseterror104-connection-reset-by-peer-or-urllib3exceptionsmaxretryerror-httpsconnectionpoolhostcdn-lfshuggingfaceco-port443) for guidelines to fix your network issue.

The issue has been fixed in the latest version of OpenCompass, so you might also consider pull from the latest version.

## Efficiency

### Why does OpenCompass partition each evaluation request into tasks?

Given the extensive evaluation time and the vast quantity of datasets, conducting a comprehensive linear evaluation on LLM models can be immensely time-consuming. To address this, OpenCompass divides the evaluation request into multiple independent "tasks". These tasks are then dispatched to various GPU groups or nodes, achieving full parallelism and maximizing the efficiency of computational resources.

### How does task partitioning work?

Each task in OpenCompass represents a combination of specific model(s) and portions of the dataset awaiting evaluation. OpenCompass offers a variety of task partitioning strategies, each tailored for different scenarios. During the inference stage, the prevalent partitioning method seeks to balance task size, or computational cost. This cost is heuristically derived from the dataset size and the type of inference.

### Why does it take more time to evaluate LLM models on OpenCompass?

There is a tradeoff between the number of tasks and the time to load the model. For example, if we partition an request that evaluates a model against a dataset into 100 tasks, the model will be loaded 100 times in total. When resources are abundant, these 100 tasks can be executed in parallel, so the additional time spent on model loading can be ignored. However, if resources are limited, these 100 tasks will operate more sequentially, and repeated loadings can become a bottleneck in execution time.

Hence, if users find that the number of tasks greatly exceeds the available GPUs, we advise setting the `--max-partition-size` to a larger value.

## Model

### How to use the downloaded huggingface models?

If you have already download the checkpoints of the model, you can specify the local path of the model. For example

```bash
python run.py --datasets siqa_gen winograd_ppl --hf-type base --hf-path /path/to/model
```

## Dataset

### How to build a new dataset?

- For building new objective dataset: [new_dataset](../advanced_guides/new_dataset.md)
- For building new subjective dataset: [subjective_evaluation](../advanced_guides/subjective_evaluation.md)
