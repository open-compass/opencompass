## Introduction

Although large-scale language models (LLMs) such as GPT-4 have demonstrated significant advantages in handling natural language tasks, most current open-source models can only handle texts with a length of a few thousand tokens, which limits their ability to process long contexts such as reading books and writing text summaries. To explore the performance of models in dealing with long contexts, we use the [L-Eval](https://github.com/OpenLMLab/LEval) and [LongBench](https://github.com/THUDM/LongBench) datasets to test the model's ability to handle long contexts.

## Existing Algorithms

When dealing with long context inputs, the two main challenges faced by large models are the inference time cost and catastrophic forgetting. Recently, a large amount of research has been devoted to extending the model length, focusing on three improvement directions:

- Attention mechanisms. The ultimate goal of these methods is to reduce the computation cost of query-key pairs, but they may affect the performance of downstream tasks.
- Input methods. Some studies divide long context inputs into chunks or retrieve pre-existing text segments to enhance the model's ability to handle long contexts, but these methods are only effective for some tasks and are difficult to adapt to multiple downstream tasks.
- Position encoding. This research includes ALiBi, xPOS, etc., which have shown good results in length extrapolation. These methods have been used to train long context models such as ChatGLM2-6B-32k and LongChat-32k.

## [L-Eval](https://github.com/OpenLMLab/LEval)

L-Eval is a long context dataset built by OpenLMLab, consisting of 18 subtasks, including texts from various fields such as law, economy, and technology. The dataset consists of a total of 411 documents, over 2000 test cases, with an average document length of 7217 words. The subtasks in this dataset are divided into close-ended and open-ended categories, with 5 close-ended tasks evaluated using the exact match criterion and 13 open-ended tasks evaluated using Rouge scores.

## [LongBench](https://github.com/THUDM/LongBench)

LongBench is a long context dataset built by THUDM, consisting of 21 subtasks with a total of 4750 test cases. This dataset is the first long context dataset that includes both English and Chinese texts, with an average English text length of 6711 words and an average Chinese text length of 13386 characters. The 21 subtasks are divided into 6 types, providing a more comprehensive evaluation of the model's capabilities in various aspects.

<div align="center">
<img src=figs/LongBenchIntro.jpg>
</div>

## Evaluation Method

Due to the different maximum input lengths accepted by different models, in order to compare these large models more fairly, when the input length exceeds the maximum input limit of the model, we will trim the middle part of the input text to avoid missing prompt words.

## Long Context Ability Ranking

In the LongBench and L-Eval ability rankings, we select the average ranking of each model in the subtask as the standard. It can be seen that GPT-4 and GPT-3.5-turbo-16k still occupy a leading position in long context tasks, while models like ChatGLM2-6B-32k also show significant improvement in long context ability after extrapolating the length based on ChatGLM2-6B.

<div align="center">
<img src=figs/L-EvalAverageRank.png>
<img src=figs/LongBenchAverageRank.png>
</div>

The original scores are available at [L-Eval Scores](/result_leval.md) and [LongBench Scores](/result_longbench.md).
