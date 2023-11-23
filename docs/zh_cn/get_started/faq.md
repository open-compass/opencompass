# 常见问题

## 通用

### OpenCompass 如何分配 GPU？

OpenCompass 使用称为 task (任务) 的单位处理评估请求。每个任务都是模型和数据集的独立组合。任务所需的 GPU 资源完全由正在评估的模型决定，具体取决于 `num_gpus` 参数。

在评估过程中，OpenCompass 部署多个工作器并行执行任务。这些工作器不断尝试获取 GPU 资源直到成功运行任务。因此，OpenCompass 始终努力充分利用所有可用的 GPU 资源。

例如，如果您在配备有 8 个 GPU 的本地机器上使用 OpenCompass，每个任务要求 4 个 GPU，那么默认情况下，OpenCompass 会使用所有 8 个 GPU 同时运行 2 个任务。但是，如果您将 `--max-num-workers` 设置为 1，那么一次只会处理一个任务，只使用 4 个 GPU。

### 为什么 HuggingFace 模型使用 GPU 的行为和我的预期不符？

这是一个比较复杂的问题，我们需要从供给和需求两侧来说明：

供给侧就是运行多少任务。任务是模型和数据集的组合，它首先取决于要测多少模型和多少数据集。另外由于 OpenCompass 会将一个较大的任务拆分成多个小任务，因此每个子任务有多少条数据 `--max-partition-size` 也会影响任务的数量。(`--max-partition-size` 与真实数据条目成正比，但并不是 1:1 的关系)。

需求侧就是有多少 worker 在运行。由于 OpenCompass 会同时实例化多个模型去进行推理，因此我们用 `--num-gpus` 来指定每个实例使用多少 GPU。注意 `--num-gpus` 是一个 HuggingFace 模型专用的参数，非 HuggingFace 模型设置该参数是不会起作用的。同时我们使用 `--max-num-workers` 去表示最多有多少个实例在运行。最后由于 GPU 显存、负载不充分等问题，OpenCompass 也支持在同一个 GPU 上运行多个实例，这个参数是 `--max-num-workers-per-gpu`。因此可以笼统地认为，我们总共会使用 `--num-gpus` * `--max-num-workers` /  `--max-num-workers-per-gpu` 个 GPU。

综上，当任务运行较慢，GPU 负载不高的时候，我们首先需要检查供给是否充足，如果不充足，可以考虑调小 `--max-partition-size` 来将任务拆分地更细；其次需要检查需求是否充足，如果不充足，可以考虑增大 `--max-num-workers` 和 `--max-num-workers-per-gpu`。一般来说，**我们会将 `--num-gpus` 设定为最小的满足需求的值，并不会再进行调整**。

### 我如何控制 OpenCompass 占用的 GPU 数量？

目前，没有直接的方法来指定 OpenCompass 可以使用的 GPU 数量。但以下是一些间接策略：

**如果在本地评估：**
您可以通过设置 `CUDA_VISIBLE_DEVICES` 环境变量来限制 OpenCompass 的 GPU 访问。例如，使用 `CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py ...` 只会向 OpenCompass 暴露前四个 GPU，确保它同时使用的 GPU 数量不超过这四个。

**如果使用 Slurm 或 DLC：**
尽管 OpenCompass 没有直接访问资源池，但您可以调整 `--max-num-workers` 参数以限制同时提交的评估任务数量。这将间接管理 OpenCompass 使用的 GPU 数量。例如，如果每个任务需要 4 个 GPU，您希望分配总共 8 个 GPU，那么应将 `--max-num-workers` 设置为 2。

### 找不到 `libGL.so.1`

opencv-python 依赖一些动态库，但环境中没有，最简单的解决办法是卸载 opencv-python 再安装 opencv-python-headless。

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

也可以根据报错提示安装对应的依赖库

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

## 网络

### 运行报错：`('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))` 或 `urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='cdn-lfs.huggingface.co', port=443)`

由于 HuggingFace 的实现，OpenCompass 在首次加载某些数据集和模型时需要网络（尤其是与 HuggingFace 的连接）。此外，每次启动时都会连接到 HuggingFace。为了成功运行，您可以：

- 通过指定环境变量 `http_proxy` 和 `https_proxy`，挂上代理；
- 使用其他机器的缓存文件。首先在有 HuggingFace 访问权限的机器上运行实验，然后将缓存文件复制到离线的机器上。缓存文件默认位于 `~/.cache/huggingface/`（[文档](https://huggingface.co/docs/datasets/cache#cache-directory)）。当缓存文件准备好时，您可以在离线模式下启动评估：
  ```python
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run.py ...
  ```
  这样，评估不再需要网络连接。但是，如果缓存中缺少任何数据集或模型的文件，仍然会引发错误。

### 我的服务器无法连接到互联网，我如何使用 OpenCompass？

如 [网络-Q1](#运行报错Connection-aborted-ConnectionResetError104-Connection-reset-by-peer-或-urllib3exceptionsMaxRetryError-HTTPSConnectionPoolhostcdn-lfshuggingfaceco-port443) 所述，使用其他机器的缓存文件。

### 在评估阶段报错 `FileNotFoundError: Couldn't find a module script at opencompass/accuracy.py. Module 'accuracy' doesn't exist on the Hugging Face Hub either.`

HuggingFace 试图将度量（例如 `accuracy`）作为在线模块加载，如果网络无法访问，它可能会失败。请参考 [网络-Q1](#运行报错Connection-aborted-ConnectionResetError104-Connection-reset-by-peer-或-urllib3exceptionsMaxRetryError-HTTPSConnectionPoolhostcdn-lfshuggingfaceco-port443) 以解决您的网络问题。

该问题在最新版 OpenCompass 中已经修复，因此也可以考虑使用最新版的 OpenCompass。

## 效率

### 为什么 OpenCompass 将每个评估请求分割成任务？

鉴于大量的评估时间和大量的数据集，对 LLM 模型进行全面的线性评估可能非常耗时。为了解决这个问题，OpenCompass 将评估请求分为多个独立的 “任务”。然后，这些任务被派发到各种 GPU 组或节点，实现全并行并最大化计算资源的效率。

### 任务分区是如何工作的？

OpenCompass 中的每个任务代表等待评估的特定模型和数据集部分的组合。OpenCompass 提供了各种任务分区策略，每种策略都针对不同的场景。在推理阶段，主要的分区方法旨在平衡任务大小或计算成本。这种成本是从数据集大小和推理类型中启发式地得出的。

### 为什么在 OpenCompass 上评估 LLM 模型需要更多时间？

任务数量与加载模型的时间之间存在权衡。例如，如果我们将评估模型与数据集的请求分成 100 个任务，模型将总共加载 100 次。当资源充足时，这 100 个任务可以并行执行，所以在模型加载上花费的额外时间可以忽略。但是，如果资源有限，这 100 个任务会更加串行地执行，重复的加载可能成为执行时间的瓶颈。

因此，如果用户发现任务数量远远超过可用的 GPU，我们建议将 `--max-partition-size` 设置为一个较大的值。

## 模型

### 如何使用本地已下好的Huggingface模型?

如果您已经提前下载好Huggingface的模型文件，请手动指定模型路径，并在`--model-kwargs` 和 `--tokenizer-kwargs`中添加 `trust_remote_code=True`. 示例如下

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
