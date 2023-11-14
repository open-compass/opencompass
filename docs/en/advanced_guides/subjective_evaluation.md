# Subjective Evaluation Guidance

## Introduction

Subjective evaluation aims to assess the model's performance in tasks that align with human preferences. The key criterion for this evaluation is human preference, but it comes with a high cost of annotation.

To explore the model's subjective capabilities, we employ state-of-the-art LLM (GPT-4) as a substitute for human assessors ([LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)).

A popular evaluation method involves comparing model responses pairwise to calculate their win rate ([Chatbot Arena](https://chat.lmsys.org/)).

We support the use of GPT-4 for the subjective evaluation of models based on this method.

## Data Preparation

We provide a demo test set [subjective_demo.xlsx](https://opencompass.openxlab.space/utils/subjective_demo.xlsx) based on [z-bench](https://github.com/zhenbench/z-bench).

Store the set of subjective questions in .xlsx format in the `data/subjective/directory`.

The table includes the following fields:

- 'question': Question description
- 'index': Question number
- 'reference_answer': Reference answer
- 'evaluating_guidance': Evaluation guidance
- 'capability': The capability dimension of the question.

## Evaluation Configuration

The specific process includes:

1. Model response reasoning
2. GPT-4 evaluation comparisons
3. Generating evaluation reports

For `config/subjective.py`, we provide some annotations to help users understand the configuration file's meaning.

```python
# Import datasets and subjective evaluation summarizer
from mmengine.config import read_base
with read_base():
    from .datasets.subjective_cmp.subjective_cmp import subjective_datasets
    from .summarizers.subjective import summarizer

datasets = [*subjective_datasets]

from opencompass.models import HuggingFaceCausalLM, HuggingFace, OpenAI

# Import partitioner and task required for subjective evaluation
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask


# Define model configurations for inference and evaluation
# Including the inference models chatglm2-6b, qwen-7b-chat, internlm-chat-7b, and the evaluation model gpt4
models = [...]

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ],
    reserved_roles=[
        dict(role='SYSTEM', api_role='SYSTEM'),
    ],
)

# Define the configuration for subjective evaluation
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        mode='all',  # alternately constructs two for comparisons
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2,  # Supports parallel comparisons
        task=dict(
            type=SubjectiveEvalTask,  # Used to read inputs for a pair of models
            judge_cfg=dict(
                abbr='GPT4',
                type=OpenAI,
                path='gpt-4-0613',
                key='ENV',
                meta_template=api_meta_template,
                query_per_second=1,
                max_out_len=2048,
                max_seq_len=2048,
                batch_size=2),
        )),
)
```

## Launching the Evaluation

```shell
python run.py config/subjective.py -r
```

The `-r` parameter allows the reuse of model inference and GPT-4 evaluation results.

## Evaluation Report

The evaluation report will be output to `output/.../summary/timestamp/report.md`, which includes win rate statistics, battle scores, and ELO ratings. The specific format is as follows:

```markdown
# Subjective Analysis

A total of 30 comparisons, of which 30 comparisons are meaningful (A / B answers inconsistent)
A total of 30 answer comparisons, successfully extracted 30 answers from GPT-4 replies, with an extraction success rate of 100.00%

### Basic statistics (4 stats: win / tie / lose / not bad)

| Dimension \ Stat [W / T / L / NB] | chatglm2-6b-hf                | qwen-7b-chat-hf              | internlm-chat-7b-hf           |
| --------------------------------- | ----------------------------- | ---------------------------- | ----------------------------- |
| LANG: Overall                     | 30.0% / 40.0% / 30.0% / 30.0% | 50.0% / 0.0% / 50.0% / 50.0% | 30.0% / 40.0% / 30.0% / 30.0% |
| LANG: CN                          | 30.0% / 40.0% / 30.0% / 30.0% | 50.0% / 0.0% / 50.0% / 50.0% | 30.0% / 40.0% / 30.0% / 30.0% |
| LANG: EN                          | N/A                           | N/A                          | N/A                           |
| CAPA: common                      | 30.0% / 40.0% / 30.0% / 30.0% | 50.0% / 0.0% / 50.0% / 50.0% | 30.0% / 40.0% / 30.0% / 30.0% |

![Capabilities Dimension Classification Result](by_capa.png)

![Language Classification  Result](by_lang.png)

### Model scores (base score is 0, win +3, both +1, neither -1, lose -3)

| Dimension \ Score | chatglm2-6b-hf | qwen-7b-chat-hf | internlm-chat-7b-hf |
| ----------------- | -------------- | --------------- | ------------------- |
| LANG: Overall     | -8             | 0               | -8                  |
| LANG: CN          | -8             | 0               | -8                  |
| LANG: EN          | N/A            | N/A             | N/A                 |
| CAPA: common      | -8             | 0               | -8                  |

### Bootstrap ELO, Median of n=1000 times

|                  | chatglm2-6b-hf | internlm-chat-7b-hf | qwen-7b-chat-hf |
| ---------------- | -------------- | ------------------- | --------------- |
| elo_score [Mean] | 999.504        | 999.912             | 1000.26         |
| elo_score [Std]  | 0.621362       | 0.400226            | 0.694434        |
```

For comparing the evaluation of models A and B, there are four choices:

1. A is better than B.
2. A and B are equally good.
3. A is worse than B.
4. Neither A nor B is good.

So, `win` / `tie` / `lose` / `not bad` represent the proportions of the model winning / tying / losing / winning or being equally good, respectively.

`Bootstrap ELO` is calculated as the median ELO score by comparing match results through 1000 random permutations.
