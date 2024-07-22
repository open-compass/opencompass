# 主观评测指引

## 介绍

主观评测旨在评估模型在符合人类偏好的能力上的表现。这种评估的黄金准则是人类喜好，但标注成本很高。

为了探究模型的主观能力，我们采用了JudgeLLM作为人类评估者的替代品（[LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)）。

流行的评估方法主要有:

- Compare模式：将模型的回答进行两两比较，以计算对战其胜率。
- Score模式：针对单模型的回答进行打分（例如：[Chatbot Arena](https://chat.lmsys.org/)）。

我们基于以上方法支持了JudgeLLM用于模型的主观能力评估（目前opencompass仓库里支持的所有模型都可以直接作为JudgeLLM进行调用，此外一些专用的JudgeLLM我们也在计划支持中）。

## 目前已支持的主观评测数据集

1. AlignBench 中文Scoring数据集（https://github.com/THUDM/AlignBench）
2. MTBench 英文Scoring数据集，两轮对话（https://github.com/lm-sys/FastChat）
3. MTBench101 英文Scoring数据集，多轮对话（https://github.com/mtbench101/mt-bench-101）
4. AlpacaEvalv2 英文Compare数据集（https://github.com/tatsu-lab/alpaca_eval）
5. ArenaHard 英文Compare数据集，主要面向coding(https://github.com/lm-sys/arena-hard/tree/main)
6. Fofo  英文Socring数据集（https://github.com/SalesforceAIResearch/FoFo/）
7. Wildbench 英文Score和Compare数据集（https://github.com/allenai/WildBench）

## 启动主观评测

类似于已有的客观评测方式，可以在configs/eval_subjective.py中进行相关配置

### 基本参数models, datasets 和 judgemodels的指定

类似于客观评测的方式，导入需要评测的models和datasets，例如

```
with read_base():
    from .datasets.subjective.alignbench.alignbench_judgeby_critiquellm import alignbench_datasets
    from .datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import subjective_datasets as alpacav2
    from .models.qwen.hf_qwen_7b import models
```

值得注意的是，由于主观评测的模型设置参数通常与客观评测不同，往往需要设置`do_sample`的方式进行推理而不是`greedy`，故可以在配置文件中自行修改相关参数，例如

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

judgemodel通常被设置为GPT4等强力模型，可以直接按照config文件中的配置填入自己的API key，或使用自定义的模型作为judgemodel

### 其他参数的指定

除了基本参数以外，还可以在config中修改`infer`和`eval`字段里的partitioner，从而设置更合适的分片方式，目前支持的分片方式主要有三种：NaivePartitoner, SizePartitioner和NumberWorkPartitioner
以及可以指定自己的workdir用以保存相关文件。

## 自定义主观数据集评测

主观评测的具体流程包括:

1. 评测数据集准备
2. 使用API模型或者开源模型进行问题答案的推理
3. 使用选定的评价模型(JudgeLLM)对模型输出进行评估
4. 对评价模型返回的预测结果进行解析并计算数值指标

### 第一步：数据准备

这一步需要准备好数据集文件以及在`Opencompass/datasets/subjective/`下实现自己数据集的类，将读取到的数据以`list of dict`的格式return

实际上可以按照自己喜欢的任意格式进行数据准备(csv, json, jsonl)等皆可，不过为了方便上手，推荐按照已有的主观数据集的格式进行构建或按照如下的json格式进行构建。
对于对战模式和打分模式，我们各提供了一个demo测试集如下：

```python
### 对战模式示例
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

### 打分模式数据集示例
[
    {
        "question": "请你扮演一个邮件管家，我让你给谁发送什么主题的邮件，你就帮我扩充好邮件正文，并打印在聊天框里。你需要根据我提供的邮件收件人以及邮件主题，来斟酌用词，并使用合适的敬语。现在请给导师发送邮件，询问他是否可以下周三下午15:00进行科研同步会，大约200字。",
        "capability": "邮件通知",
        "others": ""
    },
```

如果要准备自己的数据集，请按照以下字段进行提供，并整理为一个json文件：

- 'question'：问题描述
- 'capability'：题目所属的能力维度
- 'others'：其他可能需要对题目进行特殊处理的项目

以上三个字段是必要的，用户也可以添加其他字段，如果需要对每个问题的prompt进行单独处理，可以在'others'字段中进行一些额外设置，并在Dataset类中添加相应的字段。

### 第二步：构建评测配置

以Alignbench为例`configs/datasets/subjective/alignbench/alignbench_judgeby_critiquellm.py`，

1. 首先需要设置`subjective_reader_cfg`，用以接收从自定义的Dataset类里return回来的相关字段并指定保存文件时的output字段
2. 然后需要指定数据集的根路径`data_path`以及数据集的文件名`subjective_all_sets`，如果有多个子文件，在这个list里进行添加即可
3. 指定`subjective_infer_cfg`和`subjective_eval_cfg`，配置好相应的推理和评测的prompt
4. 最后在相应的位置指定`mode`，`summarizer`等额外信息，注意，对于不同的主观数据集，所需指定的字段可能不尽相同。此外，相应数据集的summarizer类也需要自己实现以进行数据的统计，可以参考其他数据集的summarizer实现，位于`opencompass/opencompass/summarizers/subjective`

### 第三步 启动评测并输出评测结果

```shell
python run.py configs/eval_subjective.py -r
```

- `-r` 参数支持复用模型推理和评估结果。

JudgeLLM的评测回复会保存在 `output/.../results/timestamp/xxmodel/xxdataset/.json`
评测报告则会输出到 `output/.../summary/timestamp/report.csv`。

## 主观多轮对话评测

在OpenCompass中我们同样支持了主观的多轮对话评测，以MT-Bench为例，对MTBench的评测可以参见`configs/datasets/subjective/multiround`

在多轮对话评测中，你需要将数据格式整理为如下的dialogue格式

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

值得注意的是，由于MTBench各不同的题目类型设置了不同的温度，因此我们需要将原始数据文件按照温度分成三个不同的子集以分别推理，针对不同的子集我们可以设置不同的温度，具体设置参加`configs\datasets\subjective\multiround\mtbench_single_judge_diff_temp.py`
