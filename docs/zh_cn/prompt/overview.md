# Prompt 概括

提示词 (prompt) 是 LLM 的输入，用于让 LLM 往后续写内容或计算困惑度 (ppl)，提示词的选取会对被评测模型的精度产生重大影响。如何将数据集转换为一系列的提示词的过程是由模板 (template) 来定义的。

在 opencompass 中，我们将 template 拆分为两部分：数据侧的 template 和模型侧的 template。

数据侧的 template 被称为 [prompt_template](./prompt_template.md)，它主要是来表示如何将一条数据中的多个字段转化成一个/多个提示词的过程。

模型侧的 template 被称为 [meta_template](./meta_template.md)，它主要是来表示模型是如何将这些提示词转化为模型期望的输入的过程。
