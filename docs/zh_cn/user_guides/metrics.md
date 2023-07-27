# 评估指标

在评测阶段，我们一般以数据集本身的特性来选取对应的评估，最主要的依据为**标准答案的类型**，有以下几种类型：

- **选项**：常见于分类任务，判断题以及选择题，目前这类问题的数据集占比最大，常用的有 MMLU, CEval数据集等等，评估标准一般使用准确率--`ACCEvaluator`。
- **短语**：常见于问答以及阅读理解任务，常用数据集主要包括CLUE_CMRC, CLUE_DRCD, DROP数据集等等，评估标准一般使用匹配率--EMEvaluator。
- **句子**：常见于翻译以及伪代码，命令行的生成任务中，常用的数据集主要包括Flores, Summscreen, Govrepcrs, Iwdlt2017数据集等等，评估标准一般使用BLEU(Bilingual Evaluation Understudy)--`BleuEvaluator`。
- **段落**：常见于文本摘要生成的任务，常用的数据集主要包括Lcsts, TruthfulQA, Xsum数据集等等，评估标准一般使用ROUGE（Recall-Oriented Understudy for Gisting Evaluation）--`RougeEvaluator`。
- **代码**：常见于代码生成的任务，常用的数据集主要包括Humaneval，MBPP数据集等等，评估标准一般使用执行通过率，目前Opencompass支持的有`MBPPEvaluator`、`HumanEvaluator`。

还有一类**打分类型**评测任务没有标准答案，比如评判一个模型的输出是否存在有毒，直接使用相关 API 服务进行打分，目前支持的有 `ToxicEvaluator`。

## 已支持指标

## 如何配置
