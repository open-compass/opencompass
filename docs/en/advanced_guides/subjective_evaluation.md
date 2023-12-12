# Subjective Evaluation Guidance

## Introduction

Subjective evaluation aims to assess the model's performance in tasks that align with human preferences. The key criterion for this evaluation is human preference, but it comes with a high cost of annotation.

To explore the model's subjective capabilities, we employ JudgeLLM as a substitute for human assessors ([LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)).

A popular evaluation method involves comparing model responses pairwise to calculate their win rate, another method involves calculate scores with single model response ([Chatbot Arena](https://chat.lmsys.org/)).

We support the use of GPT-4 (or other JudgeLLM) for the subjective evaluation of models based on above methods.

## Data Preparation

We provide demo test set as below:

```python
###COREV2
[
    {
        "question": "如果我在空中垂直抛球，球最初向哪个方向行进？",
        "capability": "知识-社会常识",
        "others": {
            "question": "如果我在空中垂直抛球，球最初向哪个方向行进？",
            "evaluating_guidance": "",
            "reference_answer": "上"
        }
    },...]

###CreationV0.1
[
    {
        "question": "请你扮演一个邮件管家，我让你给谁发送什么主题的邮件，你就帮我扩充好邮件正文，并打印在聊天框里。你需要根据我提供的邮件收件人以及邮件主题，来斟酌用词，并使用合适的敬语。现在请给导师发送邮件，询问他是否可以下周三下午15:00进行科研同步会，大约200字。",
        "capability": "邮件通知",
        "others": ""
    },
```

The json must includes the following fields:

- 'question': Question description
- 'capability': The capability dimension of the question.
- 'others': Other needed information.

If you want to modify prompt on each single question, you can full some other information into 'others' and construct it.

## Evaluation Configuration

The specific process includes:

1. Model response reasoning
2. JudgeLLM evaluation comparisons
3. Generating evaluation reports

### Two Model Compare Configuration

For `config/subjective_compare.py`, we provide some annotations to help users understand the configuration file's meaning.

```python
from mmengine.config import read_base
with read_base():
    from .datasets.subjective_cmp.subjective_corev2 import subjective_datasets

from opencompass.summarizers import Corev2Summarizer

datasets = [*subjective_datasets] #set dataset
models = [...] #set models to be evaluated
judge_model = [...] #set JudgeLLM

eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        mode='m2n',  #choose eval mode, in m2n mode，you need to set base_models and compare_models, it will generate the pairs between base_models and compare_models
        base_models = [...],
        compare_models = [...]
    ))

work_dir = 'Your work dir' #set your workdir, in this workdir, if you use '--reuse', it will reuse all existing results in this workdir automatically

summarizer = dict(
    type=Corev2Summarizer, #Your dataset Summarizer
    match_method='smart', #Your answer extract method
)
```

In addition, you can also change the response order of the two models, please refer to `config/subjective_compare.py`,
when `infer_order` is setting to `random`, the response will be random ordered,
when `infer_order` is setting to `double`, the response of two models will be doubled in two ways.

### Single Model Scoring Configuration

For `config/subjective_score.py`, it is mainly same with `config/subjective_compare.py`, and you just need to modify the eval mode to `singlescore`.

## Launching the Evaluation

```shell
python run.py config/subjective.py -r
```

The `-r` parameter allows the reuse of model inference and GPT-4 evaluation results.

## Evaluation Report

The response of JudgeLLM will be output to `output/.../results/timestamp/xxmodel/xxdataset/.json`.
The evaluation report will be output to `output/.../summary/timestamp/report.csv`.
