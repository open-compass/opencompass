# 长文本评测指引

## 介绍

虽然大语言模型（LLM）如GPT-4在处理自然语言任务已经展现出明显的优势，但目前的开源模型大多只能处理数千个tokens长度以内的文本，这限制了模型阅读书籍、撰写文本摘要等需要处理长文本的能力。为了探究模型在应对长文本能力时的表现，我们采用[L-Eval](https://github.com/OpenLMLab/LEval)和[LongBench](https://github.com/THUDM/LongBench)两个长文本数据集来测试模型长文本能力。

## 现有算法及模型

在处理长文本输入时，推理时间开销和灾难性遗忘是大模型面临的两大主要挑战。最近，大量研究致力于扩展模型长度，这些研究集中于以下三个改进方向。

- 注意力机制。这些方法的最终目的多为减少query-key对的计算开销，但可能对下游任务的效果产生影响。
- 输入方法。部分研究将长文本输入分块或将部分已有文本段重复输入模型以增强模型处理长文本能力，但这些方法只对部分任务有效，难以适应多种下游任务。
- 位置编码。这部分研究包括ALiBi，xPOS等，在长度外推方面展现出了良好的效果。这些方法已经被用于训练如ChatGLM2-6b-32k和LongChat-32k等长文本模型。

接下来，我们将介绍一些我们纳入评测范围的模型及其使用的算法。

### XGen-7B-8k

XGen-7B-8k是使用标准的注意力机制训练的，训练文本最长为8K，总计1.5T个token。为了减少训练时间开销, XGen-7B-8k在不同阶段逐步增加输入文本长度。首先, 模型在序列长度为2k的文本上训练总计800B的token, 随后在长度为4k的文本上训练总计400B的token, 最后, 在长度为8k的文本上训练总计300B的token。

### Vicuna-7b-v1.5-16k

Vicuna-7b-v1.5-16k使用LLaMA 2作为基座模型. 它提供了使用线性旋转式编码(RoPE)扩展的16k上下文长度。RoPE是一种在LLaMA中使用的位置编码方式，它在Transformer中更有效地注入了位置信息. 它具有一些有价值的特性，例如能够适应任何序列长度、随着相对距离的增加而减少token间的依赖性，并且能够使线性自注意力配备相对位置编码的能力。

### LongChat-7b-v1.5-32k

LongChat-7b-v1.5-32k是从LLaMA 2模型微调得到, LLaMA 2模型最初使用4k的上下文长度进行预训练。第一步是压缩RoPE。由于LLaMA 2模型在预训练阶段没有训练输入位置大于4096的token，LongChat将位置大于4096的token压缩到0到4096之间。第二步是在对话数据上微调LongChat模型。在这一步中，LongChat使用FastChat中的步骤对数据进行清洗，并将对话文本截断到模型的最大长度。

### ChatGLM2-6B-32k

ChatGLM2-6B-32k是基于ChatGLM2-6B训练的，其中在对齐和位置插值过程中使用32k的上下文长度。位置插值的关键思想是直接对位置索引进行插值，使最大位置索引与预训练阶段的上下文窗口限制相匹配。换句话说，为了容纳更多的输入token，该算法在相邻整数位置插值位置编码，利用位置编码可以应用于非整数位置，不需要在训练长度的位置之外进行外推，从而避免了不合理值的出现。该算法只需要非常短的微调周期，模型就可以完全适应大幅扩展的上下文窗口。

## [L-Eval](https://github.com/OpenLMLab/LEval)

L-Eval是由OpenLMLab构建的一个长文本数据集，由18个子任务组成，其中包含法律、经济、科技等各个领域的文本。数据集总计411篇文档，超过2000条测例，文档平均长度为7217词。该数据集将子任务划分为close-ended和open-ended两类，5个close-ended任务使用完全匹配(Exact Match)作为评测标准，而13个open-ended任务则使用Rouge分数评测。

## [LongBench](https://github.com/THUDM/LongBench)

LongBench是由THUDM构建的长文本数据集，由21个子任务构成，总计4750条测例。该数据集是第一个包含中英双语的长文本数据集，其中英语文本长度平均为6711词，中文文本平均长度为13386字。21个子任务分为以下6种类型，对模型各方面能力提供了较为全面的评测。

<div align="center">
<img src=https://github.com/open-compass/opencompass/assets/75252858/4555e937-c519-4e9c-ad8d-7370430d466a>
</div>

## 评测方法

由于不同模型能够接受的最大输入长度不同，为了更加公平地比较这些大模型，在输入长度超过模型最大输入限制时，我们将裁剪输入文本的中间部分，从而避免提示词缺失的情况。

## 长文本能力榜单

在LongBench和L-Eval能力榜单中，我们选取各模型在子任务上排名的平均值 **(排名数值越低越好)** 作为标准。可以看到GPT-4和GPT-3.5-turbo-16k在长文本任务中仍然占据领先地位，而例如ChatGLM2-6B-32k在基于ChatGLM2-6B使用位置插值后在长文本能力方面也有明显提升。

<div align="center">
<img src=https://github.com/open-compass/opencompass/assets/75252858/29b5ad12-d9a3-4255-be0a-f770923fe514>
<img src=https://github.com/open-compass/opencompass/assets/75252858/680b4cda-c2b1-45d1-8c33-196dee1a38f3>
</div>

原始分数请点击[L-Eval分数](https://github.com/open-compass/opencompass/docs/en/advanced_guides/result_leval.md)和[LongBench分数](https://github.com/open-compass/opencompass/docs/en/advanced_guides/result_longbench.md)查看。
