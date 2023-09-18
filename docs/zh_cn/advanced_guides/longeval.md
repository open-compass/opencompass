# 长文本评测指引

## 介绍

虽然大语言模型（LLM）如GPT-4在处理自然语言任务已经展现出明显的优势，但目前的开源模型大多只能处理数千个token长度以内的文本，这限制了模型阅读书籍、撰写文本摘要等需要处理长文本的能力。为了探究模型在应对长文本能力时的表现，我们采用[L-Eval](https://github.com/OpenLMLab/LEval)和[LongBench](https://github.com/THUDM/LongBench)两个长文本数据集来测试模型长文本能力。

## 现有算法及模型

在处理长文本输入时，推理时间开销和灾难性遗忘是大模型面临的两大主要挑战。最近，大量研究致力于扩展模型长度，这些研究集中于以下三个改进方向。

- 注意力机制。这些方法的最终目的多为减少query-key对的计算开销，但可能对下游任务的效果产生影响。
- 输入方法。部分研究将长文本输入分块或将部分已有文本段重复输入模型以增强模型处理长文本能力，但这些方法只对部分任务有效，难以适应多种下游任务。
- 位置编码。这部分研究包括RoPE, ALiBi，位置插值等，在长度外推方面展现出了良好的效果。这些方法已经被用于训练如ChatGLM2-6b-32k和LongChat-32k等长文本模型。

首先，我们介绍一些流行的位置编码算法。

### RoPE

RoPE是一种在Transformer中注入位置信息的位置嵌入方法。它使用旋转矩阵对绝对位置进行编码，并同时在自注意力公式中融入显式的相对位置依赖关系。下图是RoPE机制的一个示例。

<div align="center">
<img src=https://github.com/open-compass/opencompass/assets/75252858/08c57958-0dcb-40d7-b91b-33f20ca2d89f>
</div>

RoPE具有一些有价值的特性，例如可以扩展到任意序列长度、随着相对距离增加而减弱的token间依赖关系以及为线性自注意力提供相对位置编码的能力。

RoPE被应用于许多LLM模型，包括LLaMA、LLaMA 2和Vicuna-7b-v1.5-16k。

### ALiBi

尽管RoPE和其他替代原始位置编码的方法（如T5 bias）改善了外推能力，但它们的速度比原始方法慢得多，并且使用了额外的内存和参数。因此，作者引入了具有线性偏置的注意力（ALiBi）来促进高效的外推。

对于长度为L的输入子序列，注意力子层在每个head中计算第i个query $q\_{i} \\in R^{1\\times d}, (1\\leq i\\leq L)$的注意力分数，给定前i个键$K \\in R^{i\\times d}$，其中d是head维度。
$$softmax(q\_{i}K^{T})$$
ALiBi通过与相关key和query之间的距离成比例的线性递减惩罚来负向偏置注意力分数。它唯一的修改是在query-key点积之后，在其中添加了一个静态的、非学习的偏置。
$$softmax(q\_{i}K^{T}+m\\cdot\[-(i-1),...,-2,-1,0\])$$
其中m是在训练之前固定的head特定的斜率。

ALiBi去除了位置嵌入部分，它与原始位置编码方法一样快。它被用于包括mpt-7b-storywriter在内的大语言模型，该模型能够处理非常长的输入。

### 位置插值（PI）

许多现有的预训练LLM模型包括LLaMA，使用具有弱外推性质（例如RoPE）的位置编码。作者提出了一种位置插值方法，它可以轻松地实现非常长的上下文窗口，同时相对保持模型在其原始上下文窗口大小内的处理质量。

位置插值的关键思想是直接缩小位置索引，使得最大位置索引与预训练阶段的先前上下文窗口限制相匹配。换句话说，为了容纳更多的输入token，该算法在相邻的整数位置插值位置编码，利用位置编码可以应用于非整数位置的优势，它不需要在训练位置之外进行外推从而导致灾难性值的出现。该算法只需要很少的微调时间，模型就能完全适应大大扩展的上下文窗口。

下图展现了位置插值方法的机制。图中左下方说明了位置插值方法，它将位置索引（蓝色和绿色点）本身从\[0, 4096\]缩小到\[0, 2048\]，从而使它们位于预训练范围内。

<div align="center">
<img src=https://github.com/open-compass/opencompass/assets/75252858/406454ba-a811-4c66-abbe-3a5528947257>
</div>

位置插值使得基于ChatGLM2-6B的ChatGLM2-6B-32k模型能够处理32k的上下文窗口大小。

接下来，我们将介绍一些我们纳入评测范围的模型。

### XGen-7B-8k

XGen-7B-8k是使用标准的注意力机制训练的，训练文本最长为8k，总计1.5T个token。为了减少训练时间开销, XGen-7B-8k在不同阶段逐步增加输入文本长度。首先, 模型在序列长度为2k的文本上训练总计800B的token, 随后在长度为4k的文本上训练总计400B的token, 最后, 在长度为8k的文本上训练总计300B的token。

### Vicuna-7b-v1.5-16k

Vicuna-7b-v1.5-16k是从LLaMA 2微调而来的，它使用了有监督指导微调和线性RoPE扩展方法。训练数据量约为125K个对话，这些对话是从ShareGPT收集而来的。ShareGPT是一个用户可以分享他们与ChatGPT对话的网站。这些对话被打包成每个包含16k个token的序列。

### LongChat-7b-v1.5-32k

LongChat-7b-v1.5-32k也是从LLaMA 2模型微调得到, LLaMA 2模型最初使用4k的上下文长度进行预训练。LongChat-7b-v1.5-32k的第一步是压缩RoPE。由于LLaMA 2模型在预训练阶段没有训练输入位置大于4096的token，LongChat将位置大于4096的token压缩到0到4096之间。第二步是在对话数据上微调LongChat模型。在这一步中，LongChat使用FastChat中的步骤对数据进行清洗，并将对话文本截断到模型的最大长度。

### ChatGLM2-6B-32k

ChatGLM2-6B-32k进一步增强了ChatGLM2-6B的长文本能力。它采用位置插值方法，在对话对齐过程中使用32k上下文长度进行训练，因此ChatGLM2-6B-32k能够更好地处理长达32K的上下文长度。

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
