# 主观评测指引

## 介绍

主观评测旨在评估模型在符合人类偏好的能力上的表现。这种评估的黄金准则是人类喜好，但标注成本很高。

为了探究模型的主观能力，我们采用了JudgeLLM作为人类评估者的替代品（[LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)）。

流行的评估方法主要有: 1.将模型的回答进行两两比较，以计算其胜率, 2.针对单模型的回答进行打分（[Chatbot Arena](https://chat.lmsys.org/)）。

我们基于以上方法支持了JudgeLLM用于模型的主观能力评估（目前opencompass仓库里支持的所有模型都可以直接作为JudgeLLM进行调用，此外一些专用的JudgeLLM我们也在计划支持中）。

## 数据准备

对于两回答比较和单回答打分两种方法，我们各提供了一个demo测试集如下：
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

如果要准备自己的数据集，请按照以下字段进行提供，并整理为一个json文件：

- 'question'：问题描述
- 'capability'：题目所属的能力维度
- 'others'：其他可能需要对题目进行特殊处理的项目

以上三个字段是必要的，用户也可以添加其他字段，如果需要对每个问题的prompt进行单独处理，可以在'others'字段中进行一些额外设置，并在Dataset类中添加相应的字段。

## 评测配置

具体流程包括:

1. 模型回答的推理
2. JudgeLLM评估
3. 生成评测报告

### 两回答比较配置
对于两回答比较，更详细的config setting请参考 `config/subjective_compare.py`，下面我们提供了部分简略版的注释，方便用户理解配置文件的含义。

```python
from mmengine.config import read_base
with read_base():
    from .datasets.subjective_cmp.subjective_corev2 import subjective_datasets

from opencompass.summarizers import Corev2Summarizer

datasets = [*subjective_datasets] #指定需要评测的数据集
models = [...] #指定需要评测的模型
judge_model = [...] #指定JudgeLLM

eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        mode='m2n',  #选择评测模式，在m2n模式下，需要指定base_models和compare_models，将会对base_models和compare_models生成对应的两两pair（去重且不会与自身进行比较）
        base_models = [...],
        compare_models = [...]
    ))

work_dir = 'Your work dir' #指定工作目录，在此工作目录下，若使用--reuse参数启动评测，将自动复用该目录下已有的所有结果

summarizer = dict(
    type=Corev2Summarizer, #自定义数据集Summarizer
    match_method='smart' #自定义答案提取方式
)
```

### 单回答打分配置
对于单回答打分，更详细的config setting请参考 `config/subjective_score.py`，该config的大部分都与两回答比较的config相同，只需要修改评测模式即可，将评测模式设置为`singlescore`。

## 启动评测

```shell
python run.py configs/subjective_score.py -r
```

`-r` 参数支持复用模型推理和评估结果。

## 评测报告

JudgeLLM的评测回复会保存在 `output/.../results/timestamp/xxmodel/xxdataset/.json`
评测报告则会输出到 `output/.../summary/timestamp/report.csv`。
