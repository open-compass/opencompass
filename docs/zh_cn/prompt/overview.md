# Prompt 概括

提示词 (prompt) 是 LLM 的输入，用于让 LLM 往后续写内容或计算困惑度 (ppl)，提示词的选取会对被评测模型的精度产生重大影响。如何将数据集转换为一系列的提示词的过程是由模板 (template) 来定义的。

在 OpenCompass 中，我们将 template 拆分为两部分：数据侧的 template 和模型侧的 template。在测评模型时，数据会先后经过数据和模型侧的 template，最终转化为模型所需的输入。

数据侧的 template 被称为 [prompt_template](./prompt_template.md)，它表示了把数据集的字段转化成提示词的过程。

模型侧的 template 被称为 [meta_template](./meta_template.md)，它表示了模型将这些提示词转化为自身期望的输入的过程。

我们另外还提供了一些 [思维链](./chain_of_thought.md) 的 prompt 示例。
