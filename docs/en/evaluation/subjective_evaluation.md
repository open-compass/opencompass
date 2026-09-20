# Subjective Evaluation Guidance

## Introduction

Subjective evaluation aims to assess the model's performance in tasks that align with human preferences. The key criterion for this evaluation is human preference, but it comes with a high cost of annotation.

To explore the model's subjective capabilities, we employ JudgeLLM as a substitute for human assessors ([LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)).

A popular evaluation method involves

- Compare Mode: comparing model responses pairwise to calculate their win rate
- Score Mode: another method involves calculate scores with single model response ([Chatbot Arena](https://arena.ai/)).

We support the use of GPT-4 (or other JudgeLLM) for the subjective evaluation of models based on above methods.

## Currently Supported Subjective Evaluation Datasets

01. [AlignBench](https://github.com/THUDM/AlignBench) Chinese Scoring Dataset
02. [MTBench](https://github.com/lm-sys/FastChat) English Scoring Dataset, two-turn dialogue
03. [MTBench101](https://github.com/mtbench101/mt-bench-101) English Scoring Dataset, multi-turn dialogue
04. [AlpacaEvalv2](https://github.com/tatsu-lab/alpaca_eval) English Compare Dataset
05. [ArenaHard](https://github.com/lm-sys/arena-hard/tree/main) English Compare Dataset, mainly focused on coding
06. [Fofo](https://github.com/SalesforceAIResearch/FoFo/) English Scoring Dataset
07. [Wildbench](https://github.com/allenai/WildBench) English Score and Compare Dataset
08. [CompassArena](https://arena.opencompass.org.cn/) Chinese Compare Dataset
09. [CompassArena-SubjectiveBench](https://github.com/open-compass/opencompass/tree/main/opencompass/configs/datasets/subjective/compass_arena_subjective_bench) single-turn and multi-turn Compare Dataset with Bradley-Terry summarization
10. [CompassBench](https://github.com/open-compass/CompassBench) Chinese and English Compare Dataset
11. [ELBench](https://github.com/ZeroLoss-Lab/ELBench) education-focused evaluation Dataset with LLM-as-a-Judge subjective subsets
12. [FLAMES](https://github.com/AIFlames/Flames) Chinese Alignment Scoring Dataset
13. [FollowBench](https://github.com/YJiangcm/FollowBench) Chinese and English Instruction Following Scoring Dataset
14. [HelloBench](https://github.com/Quehry/HelloBench) Long Text Generation Scoring Dataset
15. [WritingBench](https://github.com/X-PLUG/WritingBench) Writing Scoring Dataset

## Initiating Subjective Evaluation

Similar to existing objective evaluation methods, you can configure related settings in `examples/eval_subjective.py`.

### Basic Parameters: Specifying models, datasets, and judgemodels

Similar to objective evaluation, import the models and datasets that need to be evaluated, for example:

```
with read_base():
    from .datasets.subjective.alignbench.alignbench_judgeby_critiquellm_rawprompt import alignbench_datasets
    from .datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import subjective_datasets as alpacav2
    from .models.openai.gpt_6_astra import models
```

### Specifying Other Parameters

In addition to the basic parameters, you can also modify the `infer` and `eval` fields in the config to set a more appropriate partitioning method. The currently supported partitioning methods mainly include three types: NaivePartitioner, SizePartitioner, and NumberWorkPartitioner. You can also specify your own workdir to save related files.

## Subjective Evaluation with Custom Dataset

The specific process includes:

1. Data preparation
2. Model response generation
3. Evaluate the response with a JudgeLLM
4. Generate JudgeLLM's response and calculate the metric

### Step-1: Data Preparation

This step requires preparing the dataset file and implementing your own dataset class under `Opencompass/datasets/subjective/`, returning the read data in the format of `list of dict`.

Actually, you can prepare the data in any format you like (csv, json, jsonl, etc.). However, to make it easier to get started, it is recommended to construct the data according to the format of the existing subjective datasets or according to the following json format.
We provide mini test-set for **Compare Mode** and **Score Mode** as below:

```python
### Compare-Mode Example
[
    {
        "question": "If I throw a ball vertically into the air, which direction does it initially travel?",
        "capability": "Knowledge - common sense",
        "others": {
            "question": "If I throw a ball vertically into the air, which direction does it initially travel?",
            "evaluating_guidance": "",
            "reference_answer": "Up"
        }
    },...]

### Score-Mode Dataset Example
[
    {
        "question": "Act as an email assistant. Draft an approximately 200-word email asking my advisor whether a research sync can be held at 15:00 next Wednesday.",
        "capability": "Email notification",
        "others": ""
    },
```

The json must includes the following fields:

- 'question': Question description
- 'capability': The capability dimension of the question.
- 'others': Other needed information.

If you want to modify prompt on each single question, you can full some other information into 'others' and construct it.

### Step-2: Evaluation Configuration(Compare Mode)

Taking Alignbench as an example, [`configs/datasets/subjective/alignbench/alignbench_judgeby_critiquellm_rawprompt.py`](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/subjective/alignbench/alignbench_judgeby_critiquellm_rawprompt.py#L1-L60):

1. First, you need to set `subjective_reader_cfg` to receive the relevant fields returned from the custom Dataset class and specify the output fields when saving files.
2. Then, you need to specify the root path `data_path` of the dataset and the dataset filename `subjective_all_sets`. If there are multiple sub-files, you can add them to this list.
3. Specify `subjective_infer_cfg` and `subjective_eval_cfg` to configure the corresponding inference and evaluation prompts.
4. Specify additional information such as `mode` at the corresponding location. Note that the fields required for different subjective datasets may vary.
5. Define post-processing and score statistics. For example, the `alignbench_postprocess` function in [`opencompass/datasets/subjective/alignbench.py`](https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/subjective/alignbench.py#L306-L318).

### Step-3: Launch the Evaluation

```shell
opencompass examples/eval_subjective.py -r
```

The `-r` parameter allows the reuse of model inference and GPT-4 evaluation results.

The response of JudgeLLM will be output to `output/.../results/timestamp/xxmodel/xxdataset/.json`.
The evaluation report will be output to `output/.../summary/timestamp/report.csv`.

## Multi-round Subjective Evaluation in OpenCompass

In OpenCompass, we also support subjective multi-turn dialogue evaluation. For instance, the evaluation of MT-Bench can be referred to in `configs/datasets/subjective/multiround`.

In the multi-turn dialogue evaluation, you need to organize the data format into the following dialogue structure:

```
"dialogue": [
    {
        "role": "user",
        "content": "Imagine you are participating in a race with a group of people. If you have just overtaken the second person, what's your current position? Where is the person you just overtook?"
    },
    {
        "role": "assistant",
        "content": ""
    },
    {
        "role": "user",
        "content": "If the \"second person\" is changed to \"last person\" in the above question, what would the answer be?"
    },
    {
        "role": "assistant",
        "content": ""
    }
],
```

It's important to note that due to the different question types in MTBench having different temperature settings, we need to divide the original data files into three different subsets according to the temperature for separate inference. For different subsets, we can set different temperatures. For specific settings, please refer to [`configs/datasets/subjective/multiround/mtbench_single_judge_diff_temp_new_dialogue.py`](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/subjective/multiround/mtbench_single_judge_diff_temp_new_dialogue.py#L1-L72).
