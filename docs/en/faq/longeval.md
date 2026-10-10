# Long Context Evaluation Guidance

## Introduction

Although large-scale language models (LLMs) such as GPT-4 have demonstrated significant advantages in handling natural language tasks, most current open-source models can only handle texts with a length of a few thousand tokens, which limits their ability to process long contexts such as reading books and writing text summaries. To explore the performance of models in dealing with long contexts, we use the [L-Eval](https://github.com/OpenLMLab/LEval) and [LongBench](https://github.com/THUDM/LongBench) datasets to test the model's ability to handle long contexts.

## Existing Algorithms and models

When dealing with long context inputs, the two main challenges faced by large models are the inference time cost and catastrophic forgetting. Recently, a large amount of research has been devoted to extending the model length, focusing on three improvement directions:

- Attention mechanisms. The ultimate goal of these methods is to reduce the computation cost of query-key pairs, but they may affect the performance of downstream tasks.
- Input methods. Some studies divide long context inputs into chunks or retrieve pre-existing text segments to enhance the model's ability to handle long contexts, but these methods are only effective for some tasks and are difficult to adapt to multiple downstream tasks.
- Position encoding. This research includes RoPE, ALiBi, Position Interpolation etc., which have shown good results in length extrapolation. These methods have been used to train long context models such as ChatGLM2-6B-32k and LongChat-32k.

First, we introduce some popular position encoding algorithms.

### RoPE

RoPE is a type of positional embedding that injects the information of position in Transformer. It encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. A graphic illustration of RoPE is shown below.

<div align="center">
<img src=https://github.com/open-compass/opencompass/assets/75252858/08c57958-0dcb-40d7-b91b-33f20ca2d89f>
</div>

RoPE comes with valuable properties such as flexibility of being expand to any sequence lengths, decaying inter-token dependency with increasing relative distances, and capability of equipping the linear self-attention with relative position encoding.

RoPE is adopted in many LLMs including LLaMA, LLaMA 2 and Vicuna-7b-v1.5-16k.

### ALiBi

Though RoPE and other alternatives to the original sinusoidal position method(like T5 bias) have improved extrapolation, they are considerably slower than the sinusoidal approach and use extra memory and parameter. Therefore, Attention with Linear Biases (ALiBi) is introduced to facilitate efficient extrapolation.

For an input subsequence of length L, the attention sublayer computes the attention scores for the ith query

```{math}
q_{i} \in R^{1 \times d}, (1 \leq i \leq L)
```

in each head, given the first i keys

```{math}
K \in R^{i \times d}
```

where d is the head dimension.

```{math}
softmax(q_{i}K^{T})
```

ALiBi negatively biases attention scores with a linearly decreasing penalty proportional to the distance between the relevant key and query.  The only modification it applies is after the query-key dot product, where it adds a static, non-learned bias.

```{math}
softmax(q_{i}K^{T}+m\cdot[-(i-1),...,-2,-1,0])
```

where scalar m is a head-specific slope fixed before training.

ALiBi eliminates position embeddings and it is as fast as the sinusoidal approach. It is used in LLMs including mpt-7b-storywriter, which is prepared to handle extremely long inputs.

### Position Interpolation(PI)

Many existing pre-trained LLMs including LLaMA use positional encodings that have weak extrapolation properties(e.g. RoPE). Position Interpolation is proposed and it can easily enable very long context windows while preserving model quality relatively well for the tasks within its original context window size.

The key idea of Position Interpolation is directly down-scale the position indices so that the maximum position index matches the previous context window limit in the pre-training stage. In other words, to accommodate more input tokens, the algorithm interpolates position encodings at neighboring integer positions, utilizing the fact that position encodings can be applied on non-integer positions, as opposed toextrapolating outside the trained positions, which may lead to catastrophic values. The algorithm requires only a very short period of fine-tuning for the model to fully adapt to greatly extended context windows.

An illustration of Position Interpolation method is shown below. Lower left illustrates Position Interpolation where it downscales the position indices (blue and green dots) themselves from \[0, 4096\] to \[0, 2048\] to force them to reside in the pretrained range.

<div align="center">
<img src=https://github.com/open-compass/opencompass/assets/75252858/406454ba-a811-4c66-abbe-3a5528947257>
</div>

Position Interpolation empowers ChatGLM2-6B-32k, a model based on ChatGLM2-6B, to deal with a 32k context window size.

Next, we introduce some long context language models we evaluate.

### XGen-7B-8k

XGen-7B-8k is trained with standard dense attention on up to 8k sequence length for up to 1.5T tokens. To mitigate slow training, XGen-7B-8k introduces training in stages with increasing sequence length. First, 800B tokens with sequence length of 2k tokens are observed, then 400B tokens with 4k, finally, 300B tokens with 8k length.

### Vicuna-7b-v1.5-16k

Vicuna-7b-v1.5-16k is fine-tuned from LLaMA 2 with supervised instruction fine-tuning and linear RoPE scaling. The training data is around 125K conversations collected from ShareGPT, a website where users can share their ChatGPT conversation. These conversations are packed into sequences that contain 16k tokens each.

### LongChat-7b-v1.5-32k

LongChat-7b-v1.5-32k is fine-tuned from LLaMA 2 models, which were originally pretrained with 4k context length. The training recipe can be conceptually described in two steps. The first step is condensing RoPE. Since the LLaMA model has not observed scenarios where position_ids > 4096 during the pre-training phase, LongChat condenses position_ids > 4096 to be within 0 to 4096. The second step is fine-tuning LongChat model on curated conversation data. In this step, the data is cleaned using FastChat data pipeline and truncated to the maximum length of model.

### ChatGLM2-6B-32k

The ChatGLM2-6B-32k further strengthens the ability to understand long texts based on the ChatGLM2-6B. Based on the method of Positional Interpolation, and trained with a 32K context length during the dialogue alignment, ChatGLM2-6B-32k can better handle up to 32K context length.

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

The original scores are shown below.

| L-Eval            | GPT-4 | GPT-3.5-turbo-16k | chatglm2-6b-32k | vicuna-7b-v1.5-16k | xgen-7b-8k | internlm-chat-7b-8k | longchat-7b-v1.5-32k | chatglm2-6b |
| ----------------- | ----- | ----------------- | --------------- | ------------------ | ---------- | ------------------- | -------------------- | ----------- |
| coursera          | 61.05 | 50                | 45.35           | 26.74              | 33.72      | 40.12               | 27.91                | 38.95       |
| gsm100            | 92    | 78                | 27              | 11                 | 8          | 19                  | 5                    | 8           |
| quality           | 81.19 | 62.87             | 44.55           | 11.39              | 33.66      | 45.54               | 29.7                 | 41.09       |
| tpo               | 72.93 | 74.72             | 56.51           | 17.47              | 44.61      | 60.59               | 17.1                 | 56.51       |
| topic_retrieval   | 100   | 79.33             | 44.67           | 24.67              | 1.33       | 0                   | 25.33                | 1.33        |
|                   |       |                   |                 |                    |            |                     |                      |             |
| financialqa       | 53.49 | 50.32             | 35.41           | 44.59              | 39.28      | 25.09               | 34.07                | 17.82       |
| gov_report        | 50.84 | 50.48             | 42.97           | 48.17              | 38.52      | 31.29               | 36.52                | 41.88       |
| legal_contract_qa | 31.23 | 27.97             | 34.21           | 24.25              | 21.36      | 19.28               | 13.32                | 17.59       |
| meeting_summ      | 31.44 | 33.54             | 29.13           | 28.52              | 27.96      | 17.56               | 22.32                | 15.98       |
| multidocqa        | 37.81 | 35.84             | 28.6            | 26.88              | 24.41      | 22.43               | 21.85                | 19.66       |
| narrativeqa       | 25.87 | 25.73             | 18.24           | 20.58              | 16.87      | 13.81               | 16.87                | 1.16        |
| nq                | 67.36 | 66.91             | 41.06           | 36.44              | 29.43      | 16.42               | 35.02                | 0.92        |
| news_summ         | 34.52 | 40.41             | 32.72           | 33.98              | 26.87      | 22.48               | 30.33                | 29.51       |
| paper_assistant   | 42.26 | 41.76             | 34.59           | 35.83              | 25.39      | 28.25               | 30.42                | 30.43       |
| patent_summ       | 48.61 | 50.62             | 46.04           | 48.87              | 46.53      | 30.3                | 41.6                 | 41.25       |
| review_summ       | 31.98 | 33.37             | 21.88           | 29.21              | 26.85      | 16.61               | 20.02                | 19.68       |
| scientificqa      | 49.76 | 48.32             | 31.27           | 31                 | 27.43      | 33.01               | 20.98                | 13.61       |
| tvshow_summ       | 34.84 | 31.36             | 23.97           | 27.88              | 26.6       | 14.55               | 25.09                | 19.45       |

| LongBench           | GPT-4 | GPT-3.5-turbo-16k | chatglm2-6b-32k | longchat-7b-v1.5-32k | vicuna-7b-v1.5-16k | internlm-chat-7b-8k | chatglm2-6b | xgen-7b-8k |
| ------------------- | ----- | ----------------- | --------------- | -------------------- | ------------------ | ------------------- | ----------- | ---------- |
| NarrativeQA         | 31.2  | 25.79             | 19.27           | 19.19                | 23.65              | 12.24               | 13.09       | 18.85      |
| Qasper              | 42.77 | 43.4              | 33.93           | 30.36                | 31.45              | 24.81               | 22.52       | 20.18      |
| MultiFieldQA-en     | 55.1  | 54.35             | 45.58           | 44.6                 | 43.38              | 25.41               | 38.09       | 37         |
| MultiFieldQA-zh     | 64.4  | 61.92             | 52.94           | 32.35                | 44.65              | 36.13               | 37.67       | 14.7       |
|                     |       |                   |                 |                      |                    |                     |             |            |
| HotpotQA            | 59.85 | 52.49             | 46.41           | 34.43                | 34.17              | 27.42               | 27.35       | 28.78      |
| 2WikiMQA            | 67.52 | 41.7              | 33.63           | 23.06                | 20.45              | 26.24               | 22.83       | 20.13      |
| Musique             | 37.53 | 27.5              | 21.57           | 12.42                | 13.92              | 9.75                | 7.26        | 11.34      |
| DuReader (zh)       | 38.65 | 29.37             | 38.53           | 20.25                | 20.42              | 11.11               | 17.18       | 8.57       |
|                     |       |                   |                 |                      |                    |                     |             |            |
| GovReport           | 32.09 | 29.92             | 32.47           | 29.83                | 29.27              | 18.38               | 22.86       | 23.37      |
| QMSum               | 24.37 | 23.67             | 23.19           | 22.71                | 23.37              | 18.45               | 21.23       | 21.12      |
| Multi_news          | 28.52 | 27.05             | 25.12           | 26.1                 | 27.83              | 24.52               | 24.7        | 23.69      |
| VCSUM (zh)          | 15.54 | 16.88             | 15.95           | 13.46                | 15.76              | 12.91               | 14.07       | 0.98       |
|                     |       |                   |                 |                      |                    |                     |             |            |
| TREC                | 78.5  | 73.5              | 30.96           | 29.23                | 32.06              | 39                  | 24.46       | 29.31      |
| TriviaQA            | 92.19 | 92.75             | 80.64           | 64.19                | 46.53              | 79.55               | 64.19       | 69.58      |
| SAMSum              | 46.32 | 43.16             | 29.49           | 25.23                | 25.23              | 43.05               | 20.22       | 16.05      |
| LSHT (zh)           | 41.5  | 34.5              | 22.75           | 20                   | 24.75              | 20.5                | 16          | 18.67      |
|                     |       |                   |                 |                      |                    |                     |             |            |
| Passage Count       | 8.5   | 3                 | 3               | 1                    | 3                  | 1.76                | 3           | 1          |
| PassageRetrieval-en | 75    | 73                | 57.5            | 20.5                 | 16.5               | 7                   | 5.5         | 12         |
| PassageRetrieval-zh | 96    | 82.5              | 58              | 15                   | 21                 | 2.29                | 5           | 3.75       |
|                     |       |                   |                 |                      |                    |                     |             |            |
| LCC                 | 59.25 | 53.49             | 53.3            | 51.46                | 49.3               | 49.32               | 46.59       | 44.1       |
| RepoBench-P         | 55.42 | 55.95             | 46.66           | 52.18                | 41.49              | 35.86               | 41.97       | 41.83      |
