# Long Context Evaluation Guidance

## Introduction

Although large-scale language models (LLMs) such as GPT-4 have demonstrated significant advantages in handling natural language tasks, most current open-source models can only handle texts with a length of a few thousand tokens, which limits their ability to process long contexts such as reading books and writing text summaries. To explore the performance of models in dealing with long contexts, we use the [L-Eval](https://github.com/OpenLMLab/LEval) and [LongBench](https://github.com/THUDM/LongBench) datasets to test the model's ability to handle long contexts.

## Existing Algorithms and models

When dealing with long context inputs, the two main challenges faced by large models are the inference time cost and catastrophic forgetting. Recently, a large amount of research has been devoted to extending the model length, focusing on three improvement directions:

- Attention mechanisms. The ultimate goal of these methods is to reduce the computation cost of query-key pairs, but they may affect the performance of downstream tasks.
- Input methods. Some studies divide long context inputs into chunks or retrieve pre-existing text segments to enhance the model's ability to handle long contexts, but these methods are only effective for some tasks and are difficult to adapt to multiple downstream tasks.
- Position encoding. This research includes ALiBi, xPOS, etc., which have shown good results in length extrapolation. These methods have been used to train long context models such as ChatGLM2-6B-32k and LongChat-32k.

Next, we will introduce some algorithms adopted by long context language models we evaluate.

### XGen-7B-8k

XGen-7B-8k is trained with standard dense attention on up to 8k sequence length for up to 1.5T tokens. To mitigate slow training, XGen-7B-8k introduces training in stages with increasing sequence length. First, 800B tokens with sequence length of 2k tokens are observed, then 400B tokens with 4k, finally, 300B tokens with 8k length.

### Vicuna-7b-v1.5-16k

Vicuna-7b-v1.5-16k uses LLaMA 2 as the base model. It provides 16K context length versions using linear rotary position embedding(RoPE) scaling. RoPE is a type of positional embedding used in LLaMA that injects the information of position in Transformer. It comes with valuable properties such as flexibility of being expand to any sequence lengths, decaying inter-token dependency with increasing relative distances, and capability of equipping the linear self-attention with relative position encoding.

### LongChat-7b-v1.5-32k

LongChat-7b-v1.5-32k is finetuned from LLaMA 2 models, which were originally pretrained with 4k context length. The training recipe can be conceptually described in two steps. The first step is condensing RoPE. Since the LLaMA model has not observed scenarios where position_ids > 4096 during the pre-training phase, LongChat condenses position_ids > 4096 to be within 0 to 4096. The second step is finetuning LongChat model on curated conversation data. In this step, the data is cleaned using FastChat data pipeline and truncated to the maximum length of model.

### ChatGLM2-6B-32k

ChatGLM2-6B-32k is trained based on ChatGLM2-6B, with a 32k context length during alignment and position interpolation. The key idea of position interpolation is directly down-scale the position indices so that the maximum position index matches the previous context window limit in the pre-training stage. In other words, to accommodate more input tokens, the algorithm interpolates position encodings at neighboring integer positions,utilizing the fact that position encodings can be applied on non-integer positions, as opposed to extrapolating outside the trained positions, which may lead to catastrophic values. The algorithm requires only a very short period of fine-tuning for the model to fully adapt to greatly extended context windows.

## [L-Eval](https://github.com/OpenLMLab/LEval)

L-Eval is a long context dataset built by OpenLMLab, consisting of 18 subtasks, including texts from various fields such as law, economy, and technology. The dataset consists of a total of 411 documents, over 2000 test cases, with an average document length of 7217 words. The subtasks in this dataset are divided into close-ended and open-ended categories, with 5 close-ended tasks evaluated using the exact match criterion and 13 open-ended tasks evaluated using Rouge scores.

## [LongBench](https://github.com/THUDM/LongBench)

LongBench is a long context dataset built by THUDM, consisting of 21 subtasks with a total of 4750 test cases. This dataset is the first long context dataset that includes both English and Chinese texts, with an average English text length of 6711 words and an average Chinese text length of 13386 characters. The 21 subtasks are divided into 6 types, providing a more comprehensive evaluation of the model's capabilities in various aspects.

<div align="center">
<img src=https://github.com/open-compass/opencompass/assets/75252858/4555e937-c519-4e9c-ad8d-7370430d466a>
</div>

## Evaluation Method

Due to the different maximum input lengths accepted by different models, in order to compare these large models more fairly, when the input length exceeds the maximum input limit of the model, we will trim the middle part of the input text to avoid missing prompt words.

## Long Context Ability Ranking

In the LongBench and L-Eval ability rankings, we select the average ranking **(The lower the better)** of each model in the subtask as the standard. It can be seen that GPT-4 and GPT-3.5-turbo-16k still occupy a leading position in long context tasks, while models like ChatGLM2-6B-32k also show significant improvement in long context ability after position interpolation based on ChatGLM2-6B.

<div align="center">
<img src=https://github.com/open-compass/opencompass/assets/75252858/29b5ad12-d9a3-4255-be0a-f770923fe514>
<img src=https://github.com/open-compass/opencompass/assets/75252858/680b4cda-c2b1-45d1-8c33-196dee1a38f3>
</div>

The original scores are available at [L-Eval Scores](https://github.com/open-compass/opencompass/docs/en/advanced_guides/result_leval.md) and [LongBench Scores](https://github.com/open-compass/opencompass/docs/en/advanced_guides/result_longbench.md).
