# Subjective Evaluation Guidance

## Introduction

Subjective evaluation aims to assess the model's performance in tasks that align with human preferences. The key criterion for this evaluation is human preference, but it comes with a high cost of annotation.

To explore the model's subjective capabilities, we employ JudgeLLM as a substitute for human assessors ([LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)).

A popular evaluation method involves

- Compare Mode: comparing model responses pairwise to calculate their win rate
- Score Mode: another method involves calculate scores with single model response ([Chatbot Arena](https://chat.lmsys.org/)).

We support the use of GPT-4 (or other JudgeLLM) for the subjective evaluation of models based on above methods.

## Currently Supported Subjective Evaluation Datasets

1. AlignBench Chinese Scoring Dataset (https://github.com/THUDM/AlignBench)
2. MTBench English Scoring Dataset, two-turn dialogue (https://github.com/lm-sys/FastChat)
3. MTBench101 English Scoring Dataset, multi-turn dialogue (https://github.com/mtbench101/mt-bench-101)
4. AlpacaEvalv2 English Compare Dataset (https://github.com/tatsu-lab/alpaca_eval)
5. ArenaHard English Compare Dataset, mainly focused on coding (https://github.com/lm-sys/arena-hard/tree/main)
6. Fofo English Scoring Dataset (https://github.com/SalesforceAIResearch/FoFo/)
7. Wildbench English Score and Compare Dataset（https://github.com/allenai/WildBench）

## Initiating Subjective Evaluation

Similar to existing objective evaluation methods, you can configure related settings in `configs/eval_subjective.py`.

### Basic Parameters: Specifying models, datasets, and judgemodels

Similar to objective evaluation, import the models and datasets that need to be evaluated, for example:

```
with read_base():
    from .datasets.subjective.alignbench.alignbench_judgeby_critiquellm import alignbench_datasets
    from .datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import subjective_datasets as alpacav2
    from .models.qwen.hf_qwen_7b import models
```

It is worth noting that since the model setup parameters for subjective evaluation are often different from those for objective evaluation, it often requires setting up `do_sample` for inference instead of `greedy`. You can modify the relevant parameters in the configuration file as needed, for example:

```
models = [
    dict(
        type=HuggingFaceChatGLM3,
        abbr='chatglm3-6b-hf2',
        path='THUDM/chatglm3-6b',
        tokenizer_path='THUDM/chatglm3-6b',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        generation_kwargs=dict(
            do_sample=True,
        ),
        meta_template=api_meta_template,
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
```

The judgemodel is usually set to a powerful model like GPT4, and you can directly enter your API key according to the configuration in the config file, or use a custom model as the judgemodel.

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

### Step-2: Evaluation Configuration(Compare Mode)

Taking Alignbench as an example, `configs/datasets/subjective/alignbench/alignbench_judgeby_critiquellm.py`:

1. First, you need to set `subjective_reader_cfg` to receive the relevant fields returned from the custom Dataset class and specify the output fields when saving files.
2. Then, you need to specify the root path `data_path` of the dataset and the dataset filename `subjective_all_sets`. If there are multiple sub-files, you can add them to this list.
3. Specify `subjective_infer_cfg` and `subjective_eval_cfg` to configure the corresponding inference and evaluation prompts.
4. Finally, specify additional information such as `mode`, `summarizer`, etc., at the appropriate location. Note that for different subjective datasets, the fields that need to be specified may vary. Additionally, the summarizer class for the respective dataset also needs to be implemented to perform data statistics. You can refer to the summarizer implementations of other datasets, located in `opencompass/opencompass/summarizers/subjective`.

### Step-3: Launch the Evaluation

```shell
python run.py config/eval_subjective_score.py -r
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

It's important to note that due to the different question types in MTBench having different temperature settings, we need to divide the original data files into three different subsets according to the temperature for separate inference. For different subsets, we can set different temperatures. For specific settings, please refer to `configs\datasets\subjective\multiround\mtbench_single_judge_diff_temp.py`.
