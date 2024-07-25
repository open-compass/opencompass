# 常见问题

## 通用

### OpenCompass 为什么有这么多 bug?

OpenCompass 在开发团队中是有内部和外部两个版本，开发团队的第一优先级是保证内部版本的功能正确，对于外部的版本会相对有所疏忽。加上开发团队人力有限，水平有限，项目中因此会有很多的问题，恳请大家多多包涵。

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

### OpenCompass task 的默认划分逻辑是什么样的？

OpenCompass 默认使用 num_worker_partitioner。OpenCompass 的评测从本质上来说就是有一系列的模型和一系列的数据集，然后两两组合，用每个模型去跑每个数据集。对于同一个模型，OpenCompass 会将其拆分为 `--max-num-workers` (或 config 中的 `infer.runner.max_num_workers`) 个 task，为了保证每个 task 的运行耗时均匀，每个 task 均会所有数据集的一部分。示意图如下：

![num_worker_partitioner](https://github.com/open-compass/opencompass/assets/17680578/68c57a57-0804-4865-a0c6-133e1657b9fc)

### OpenCompass 在 slurm 等方式运行时，为什么会有部分 infer log 不存在？

因为 log 的文件名并不是一个数据集的切分，而是一个 task 的名字。由于 partitioner 有可能会将多个较小的任务合并成一个大的，而 task 的名字往往就是第一个数据集的名字。因此该 task 中后面的数据集的名字对应的 log 都不会出现，而是会直接写在第一个数据集对应的 log 中

### OpenCompass 中的断点继续逻辑是什么样的？

只要使用 --reuse / -r 开关，则会进行断点继续。首先 OpenCompass 会按照最新的 config 文件配置模型和数据集，然后在 partitioner 确定分片大小并切片后，对每个分片依次查找，若该分片已完成，则跳过；若该分片未完成或未启动，则加入待测列表。然后将待测列表中的任务依次进行执行。注意，未完成的任务对应的输出文件名一般是 `tmp_xxx`，此时模型会从该文件中标号最大的一个数据开始往后继续跑，直到完成这个分片。

根据上述过程，有如下推论：

- 在已有输出文件夹的基础上断点继续时，不可以更换 partitioner 的切片方式，或者不可以修改 `--max-num-workers` 的入参。(除非使用了 `tools/prediction_merger.py` 工具)
- 如果数据集有了任何修改，不要断点继续，或者根据需要将原有的输出文件进行删除后，全部重跑。

### OpenCompass 如何分配 GPU？

OpenCompass 使用称为 task (任务) 的单位处理评估请求。每个任务都是模型和数据集的独立组合。任务所需的 GPU 资源完全由正在评估的模型决定，具体取决于 `num_gpus` 参数。

在评估过程中，OpenCompass 部署多个工作器并行执行任务。这些工作器不断尝试获取 GPU 资源直到成功运行任务。因此，OpenCompass 始终努力充分利用所有可用的 GPU 资源。

例如，如果您在配备有 8 个 GPU 的本地机器上使用 OpenCompass，每个任务要求 4 个 GPU，那么默认情况下，OpenCompass 会使用所有 8 个 GPU 同时运行 2 个任务。但是，如果您将 `--max-num-workers` 设置为 1，那么一次只会处理一个任务，只使用 4 个 GPU。

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

### 运行报错 Error: mkl-service + Intel(R) MKL

报错全文如下：

```text
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp-a34b3233.so.1 library.
	Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
```

可以通过设置环境变量 `MKL_SERVICE_FORCE_INTEL=1` 来解决这个问题。

## 网络

### 运行报错：`('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))` 或 `urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='cdn-lfs.huggingface.co', port=443)`

由于 HuggingFace 的实现，OpenCompass 在首次加载某些数据集和模型时需要网络（尤其是与 HuggingFace 的连接）。此外，每次启动时都会连接到 HuggingFace。为了成功运行，您可以：

- 通过指定环境变量 `http_proxy` 和 `https_proxy`，挂上代理；
- 使用其他机器的缓存文件。首先在有 HuggingFace 访问权限的机器上运行实验，然后将缓存文件复制 / 软链到离线的机器上。缓存文件默认位于 `~/.cache/huggingface/`（[文档](https://huggingface.co/docs/datasets/cache#cache-directory)）。当缓存文件准备好时，您可以在离线模式下启动评估：
  ```python
  HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 HF_HUB_OFFLINE=1 python run.py ...
  ```
  这样，评估不再需要网络连接。但是，如果缓存中缺少任何数据集或模型的文件，仍然会引发错误。
- 使用中国大陆内的镜像源，例如 [hf-mirror](https://hf-mirror.com/)
  ```python
  HF_ENDPOINT=https://hf-mirror.com python run.py ...
  ```

### 我的服务器无法连接到互联网，我如何使用 OpenCompass？

如 [网络-Q1](#运行报错Connection-aborted-ConnectionResetError104-Connection-reset-by-peer-或-urllib3exceptionsMaxRetryError-HTTPSConnectionPoolhostcdn-lfshuggingfaceco-port443) 所述，使用其他机器的缓存文件。

## 效率

### 为什么 OpenCompass 将每个评估请求分割成任务？

鉴于大量的评估时间和大量的数据集，对 LLM 模型进行全面的线性评估可能非常耗时。为了解决这个问题，OpenCompass 将评估请求分为多个独立的 “任务”。然后，这些任务被派发到各种 GPU 组或节点，实现全并行并最大化计算资源的效率。

### 任务分区是如何工作的？

OpenCompass 中的每个任务代表等待评估的特定模型和数据集部分的组合。OpenCompass 提供了各种任务分区策略，每种策略都针对不同的场景。在推理阶段，主要的分区方法旨在平衡任务大小或计算成本。这种成本是从数据集大小和推理类型中启发式地得出的。

### 为什么在 OpenCompass 上评估 LLM 模型需要更多时间？

请检查：

1. 是否有使用 vllm / lmdeploy 等推理后端，这会大大提速测试过程
2. 对于使用原生 huggingface 跑的，`batch_size` 为 1 会大幅拖慢测试过程，可以适当调大 `batch_size`
3. 如果是 huggingface 上下载的模型，是否有大量的时间卡在网络连接或模型下载上面了
4. 模型的推理结果是否会意外地长，尤其是模型是否在尝试再出若干题并尝试进行解答，这在基座模型中会尤其常见。可以通过在数据集中添加 `stopping_criteria` 的方式来解决

如果上述检查项没有解决问题，请考虑给我们报 bug

## 模型

### 如何使用本地已下好的 Huggingface 模型?

如果您已经提前下载好 Huggingface 的模型文件，请手动指定模型路径. 示例如下

```bash
python run.py --datasets siqa_gen winograd_ppl --hf-type base --hf-path /path/to/model
```

## 数据集

### 如何构建自己的评测数据集

- 客观数据集构建参见：[支持新数据集](../advanced_guides/new_dataset.md)
- 主观数据集构建参见：[主观评测指引](../advanced_guides/subjective_evaluation.md)
