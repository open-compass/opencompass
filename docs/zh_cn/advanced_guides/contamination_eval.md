# 污染评估指南

**数据污染**，即下游任务的测试数据存在于大型语言模型（LLMs）的预训练数据中，可能会夸大在许多下游任务（例如，摘要、自然语言推理、文本分类）上观察到的LLM性能。

为了评估LLM在污染数据下的性能，我们使用了[Contamination Detector](https://github.com/liyucheng09/Contamination_Detector)来生成污染标签。

## [检测工具](https://github.com/liyucheng09/Contamination_Detector)简介

污染检测器有助于在不需要访问LLM的训练数据的情况下，基于互联网存在验证，识别和分析此类潜在污染，使得即使是小团队和个人也能进行强大的评估。

### 方法

- 使用必应搜索API检查逐字测试样例是否在线出现，这可能表明其包含在Common Crawl中。

- 具体来说，是通过仅搜索URL而不是完整内容，来验证包含逐字测试样例的页面是否在2017-2020年的Common Crawl中被索引。

#### 构造查询

例如：
**问题**：The flaw in Anderson’s ACT theory was that some considered it \_\_\_\_.
**选项**：
A: ’Only applicable to a motor system’,
B: ’Untestable and thus, of uncertain sci-
entific value’,
C: ’Lacking in definition for its ele-
ments’
D: ’Overly complex in explaining the
operation of cognition’,
**答案**：B
**查询**：The flaw in Anderson’s ACT theory was that some considered it untestable and thus, of uncertain scientific value.

#### 提高匹配度

为避免可能的误报，该方法配置了两个关键设置：

- 用于METEOR的排序罚分（gamma为0.8）确保匹配遵循序列；
- 匹配被限制在最多2倍查询长度的窗口内，防止部分或脱离上下文的匹配。

#### 污染类型

- *input contamination*，其中只有问题出现在匹配页面中，但没有答案；
- *input-and-label contamination*，其中问题和答案都出现在匹配页面中。

## 数据准备

待完成

## 评估配置

待完成
