# 常见问题

## 通用

### ppl 和 gen 有什么区别和联系？

`ppl` 是困惑度 (perplexity) 的缩写，是一种评价模型进行语言建模能力的指标。在 OpenCompass 的语境下，它一般指一种选择题的做法：给定一个上下文，模型需要从多个备选项中选择一个最合适的。此时，我们会将 n 个选项拼接上上下文后，形成 n 个序列，然后计算模型对这 n 个序列的 perplexity，我们认为其中 perplexity 最低的序列所对应的选项即为模型在这道题上面的推理结果，该种评测方法的后处理简单直接、确定性高。

`gen` 是生成 (generate) 的缩写。在 OpenCompass 的语境下，它指的是在给定上下文的情况下，模型往后续写的结果就是这道题目上的推理结果。一般来说，续写得到的字符串需要结合上比较重的后处理过程，才能进行可靠的答案提取，从而完成评测。

从使用上来说，基座模型的单项选择题和部分具有选择题性质的题目会使用 `ppl`，基座模型的不定项选择和非选择题都会使用 `gen`。而对话模型的所有题目都会使用 `gen`，因为许多商用 API 模型不会暴露 `ppl` 的接口。但也存在例外情况，例如我们希望基座模型输出解题思路过程时 (例如 Let's think step by step)，我们同样会使用 `gen`，但总体的使用如下图所示：

|          | ppl         | gen                |
| -------- | ----------- | ------------------ |
| 基座模型 | 仅 MCQ 任务 | MCQ 以外的其他任务 |
| 对话模型 | 无          | 所有任务           |

与 `ppl` 高度类似地，条件对数概率 `clp` (conditional log probability) 是在给定上下文的情况下，计算下一个 token 的概率。它也仅适用于选择题，考察概率的范围仅限于备选项标号所对应的 token，取其中概率最高的 token 所对应的选项为模型的推理结果。与 ppl 相比，`clp` 的计算更加高效，仅需要推理一次，而 ppl 需要推理 n 次，但坏处是，`clp` 受制于 tokenizer，在例如选项前后有无空格符号时，tokenizer 编码的结果会有变化，导致测试结果不可靠。因此 OpenCompass 中很少使用 `clp`。

### OpenCompass 如何控制 few shot 评测的 shot 数目？

在数据集配置文件中，有一个 `retriever` 的字段，该字段表示如何召回数据集中的样本作为上下文样例，其中最常用的是 `FixKRetriever` 表示固定使用某 k 个样本，因此即为 k-shot。另外还有 `ZeroRetriever` 表示不使用任何样本，这在大多数情况下意味着 0-shot。

另一方面，in context 的样本也可以直接在数据集的模板中指定，在该情况下亦会搭配使用 `ZeroRetriever`，但此时的评测并不是 0-shot，而需要根据具体的模板来进行确定。具体请看 [prompt](../prompt/prompt_template.md)

### OpenCompass 如何分配 GPU？

OpenCompass 使用称为 task (任务) 的单位处理评估请求。每个任务都是模型和数据集的独立组合。任务所需的 GPU 资源完全由正在评估的模型决定，具体取决于 `num_gpus` 参数。

在评估过程中，OpenCompass 部署多个工作器并行执行任务。这些工作器不断尝试获取 GPU 资源直到成功运行任务。因此，OpenCompass 始终努力充分利用所有可用的 GPU 资源。

例如，如果您在配备有 8 个 GPU 的本地机器上使用 OpenCompass，每个任务要求 4 个 GPU，那么默认情况下，OpenCompass 会使用所有 8 个 GPU 同时运行 2 个任务。但是，如果您将 `--max-num-workers` 设置为 1，那么一次只会处理一个任务，只使用 4 个 GPU。

### 为什么 HuggingFace 模型使用 GPU 的行为和我的预期不符？

这是一个比较复杂的问题，我们需要从供给和需求两侧来说明：

供给侧就是运行多少任务。任务是模型和数据集的组合，它首先取决于要测多少模型和多少数据集。另外由于 OpenCompass 会将一个较大的任务拆分成多个小任务，因此每个子任务有多少条数据 `--max-partition-size` 也会影响任务的数量。(`--max-partition-size` 与真实数据条目成正比，但并不是 1:1 的关系)。

需求侧就是有多少 worker 在运行。由于 OpenCompass 会同时实例化多个模型去进行推理，因此我们用 `--hf-num-gpus` 来指定每个实例使用多少 GPU。注意 `--hf-num-gpus` 是一个 HuggingFace 模型专用的参数，非 HuggingFace 模型设置该参数是不会起作用的。同时我们使用 `--max-num-workers` 去表示最多有多少个实例在运行。最后由于 GPU 显存、负载不充分等问题，OpenCompass 也支持在同一个 GPU 上运行多个实例，这个参数是 `--max-num-workers-per-gpu`。因此可以笼统地认为，我们总共会使用 `--hf-num-gpus` * `--max-num-workers` / `--max-num-workers-per-gpu` 个 GPU。

综上，当任务运行较慢，GPU 负载不高的时候，我们首先需要检查供给是否充足，如果不充足，可以考虑调小 `--max-partition-size` 来将任务拆分地更细；其次需要检查需求是否充足，如果不充足，可以考虑增大 `--max-num-workers` 和 `--max-num-workers-per-gpu`。一般来说，**我们会将 `--hf-num-gpus` 设定为最小的满足需求的值，并不会再进行调整**。

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
- 使用中国大陆内的镜像源，例如 [hf-mirror](https://hf-mirror.com/)
  ```python
  HF_ENDPOINT=https://hf-mirror.com python run.py ...
  ```

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

### 如何使用本地已下好的 Huggingface 模型?

如果您已经提前下载好 Huggingface 的模型文件，请手动指定模型路径. 示例如下

```bash
python run.py --datasets siqa_gen winograd_ppl --hf-type base --hf-path /path/to/model
```
