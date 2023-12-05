# FAQ

## General

### How does OpenCompass allocate GPUs?

OpenCompass processes evaluation requests using the unit termed as "task". Each task is an independent combination of model(s) and dataset(s). The GPU resources needed for a task are determined entirely by the model being evaluated, specifically by the `num_gpus` parameter.

During evaluation, OpenCompass deploys multiple workers to execute tasks in parallel. These workers continuously try to secure GPU resources and run tasks until they succeed. As a result, OpenCompass always strives to leverage all available GPU resources to their maximum capacity.

For instance, if you're using OpenCompass on a local machine equipped with 8 GPUs, and each task demands 4 GPUs, then by default, OpenCompass will employ all 8 GPUs to concurrently run 2 tasks. However, if you adjust the `--max-num-workers` setting to 1, then only one task will be processed at a time, utilizing just 4 GPUs.

### Why doesn't the GPU behavior of HuggingFace models align with my expectations?

This is a complex issue that needs to be explained from both the supply and demand sides:

The supply side refers to how many tasks are being run. A task is a combination of a model and a dataset, and it primarily depends on how many models and datasets need to be tested. Additionally, since OpenCompass splits a larger task into multiple smaller tasks, the number of data entries per sub-task (`--max-partition-size`) also affects the number of tasks. (The `--max-partition-size` is proportional to the actual number of data entries, but the relationship is not 1:1).

The demand side refers to how many workers are running. Since OpenCompass instantiates multiple models for inference simultaneously, we use `--num-gpus` to specify how many GPUs each instance uses. Note that `--num-gpus` is a parameter specific to HuggingFace models and setting this parameter for non-HuggingFace models will not have any effect. We also use `--max-num-workers` to indicate the maximum number of instances running at the same time. Lastly, due to issues like GPU memory and insufficient load, OpenCompass also supports running multiple instances on the same GPU, which is managed by the parameter `--max-num-workers-per-gpu`. Therefore, it can be generally assumed that we will use a total of `--num-gpus` * `--max-num-workers` / `--max-num-workers-per-gpu` GPUs.

In summary, when tasks run slowly or the GPU load is low, we first need to check if the supply is sufficient. If not, consider reducing `--max-partition-size` to split the tasks into finer parts. Next, we need to check if the demand is sufficient. If not, consider increasing `--max-num-workers` and `--max-num-workers-per-gpu`. Generally, **we set `--num-gpus` to the minimum value that meets the demand and do not adjust it further.**

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

If you have already download the checkpoints of the model, you can specify the local path of the model and tokenizer, and add `trust_remote_code=True` for `--model-kwargs` and `--tokenizer-kwargs`. For example

```bash
python run.py --datasets siqa_gen winograd_ppl \
--hf-path /path/to/model \  # HuggingFace 模型地址
--tokenizer-path /path/to/model \  # HuggingFace 模型地址
--model-kwargs device_map='auto' trust_remote_code=True \  # 构造 model 的参数
--tokenizer-kwargs padding_side='left' truncation='left' use_fast=False trust_remote_code=True \  # 构造 tokenizer 的参数
--max-out-len 100 \  # 模型能接受的最大序列长度
--max-seq-len 2048 \  # 最长生成 token 数
--batch-size 8 \  # 批次大小
--no-batch-padding \  # 不打开 batch padding，通过 for loop 推理，避免精度损失
--num-gpus 1  # 所需 gpu 数
```
