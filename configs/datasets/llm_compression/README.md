# LLM Compression

## Introduction

The following introduction comes from the abstract of [Compression Represents Intelligence Linearly](https://arxiv.org/abs/2404.09937):

>There is a belief that learning to compress well will lead to intelligence. Recently, language modeling has been shown to be equivalent to compression, which offers a compelling rationale for the success of large language models (LLMs): the development of more advanced language models is essentially enhancing compression which facilitates intelligence. ...our findings suggest that compression efficiency, as an unsupervised metric derived from raw text corpora, serves as a reliable evaluation measure that is linearly associated with the model capabilities. We open-source our compression datasets as well as our data collection pipelines to facilitate future researchers to assess compression properly.


## Official Links

- Paper: [Compression Represents Intelligence Linearly](https://arxiv.org/abs/2404.09937)
- GitHub Repository: [llm-compression-intelligence](https://github.com/hkust-nlp/llm-compression-intelligence)


## Overview and Usage

### Dataset
The dataset, which consists of three external corpora, can be downloaded using the following python script:

```python
from os import os.path as osp
from datasets import load_dataset

data_path = "data/llm-compression"

subset_mapping = {
    'arxiv_math': ['arxiv_math'],
    'commoncraw': ['cc'],
    'python': ['python'],
}

for key, value in subset_mapping.items():
    llmc_dataset = load_dataset(r"hkust-nlp/llm-compression", name=value)
    llmc_dataset["test"].to_json(osp.join(data_path, f"{key}.jsonl"))
```

Note: Refer to the original [repository](https://github.com/hkust-nlp/llm-compression-intelligence) for more details on data collection and design.


### Inference

The inference stage (`SWCELossInferencer`) consists of the following key steps:

1. For each candidate model, we obtain the encodings of each sample of the dataset using its tokenizer.
2. Concatenate the encodings of all samples into a single array and construct a PyTorch Dataset. Each item of `__getitem__` is a chunk of the array based on a sliding window. To reproduce results from the original paper, set `block_size=1900` and `stride=512`.
3. For each batch, calculate the cross entropy loss based on model logits and targets. The losses within each batch is reduced to a single loss by summation.
4. Output the losses and `total_chr_num` to `BPCEvaluator` for evaluation.


### Evaluation

`BPCEvaluator`: Using the total loss for each batch and the total number of characters in the original dataset from the inference stage, calculate the Bits per Character (BPC) metric for each model:

$$ BPC = \frac{TotalCrossEntropyLoss}{TotalCharacterNumber*log(2)} $$


### Summarization



### Config Files

1. Dataset config: `configs/datasets/llm-compression.py`
2. Evaluation config: `configs/eval_llm_compression.py`

## Evaluation Results
```
   metric version            model commoncraw  python arxiv_math  average
0     bpc  af04af   qwen1.5-32b-hf     0.5910  0.2584     0.4080   0.4191
1     bpc  af04af   qwen1.5-14b-hf     0.6459  0.2766     0.4310   0.4512
2     bpc  af04af      qwen-14b-hf     0.6197  0.2849     0.4498   0.4515
3     bpc  af04af     llama-30b-hf     0.5773  0.3212     0.4562   0.4516
4     bpc  af04af   llama-2-13b-hf     0.5807  0.3336     0.4752   0.4632
5     bpc  af04af    qwen1.5-7b-hf     0.6658  0.2935     0.4500   0.4698
6     bpc  af04af       qwen-7b-hf     0.6453  0.3088     0.4830   0.4790
7     bpc  af04af     llama-13b-hf     0.6083  0.3555     0.4865   0.4834
8     bpc  af04af    llama-2-7b-hf     0.6117  0.3536     0.4995   0.4883
9     bpc  af04af      llama-7b-hf     0.6285  0.3794     0.5096   0.5058
10    bpc  af04af  qwen1.5-1.8b-hf     0.7448  0.4029     0.5625   0.5701
11    bpc  af04af     qwen-1.8b-hf     0.7542  0.4175     0.5842   0.5853
12    bpc  af04af  qwen1.5-0.5b-hf     0.8102  0.4520     0.6181   0.6268
```


## FAQ

### I am getting this warning during inference, should I truncate long samples to `max_seq_len` to avoid further errors?
```
Token indices sequence length is longer than the specified maximum sequence length for this model. Running this sequence through the model will result in indexing errors
```
>A: This warning comes from the tokenizer indicating that the input sequence length exceeds the model's input length, but it does not affect the operation of the tokenizer. For loss calculation, as long as we set a `block_size` of the sliding window less than `max_seq_len`, we can safely ignore this warning.


## Reference
```
@misc{huang2024compression,
      title={Compression Represents Intelligence Linearly},
      author={Yuzhen Huang and Jinghan Zhang and Zifei Shan and Junxian He},
      year={2024},
      eprint={2404.09937},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
