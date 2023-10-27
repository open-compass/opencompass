# 主观评测指引

## 介绍

主观评测旨在评估模型在符合人类偏好的能力上的表现。这种评估的黄金准则是人类喜好，但标注成本很高。

为了探究模型的主观能力，我们采用了最先进的 LLM（GPT-4）作为人类评估者的替代品（[LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)）。

流行的评估方法是将模型的回答进行两两比较，以计算其胜率（[Chatbot Arena](https://chat.lmsys.org/)）。

我们基于这一方法支持了 GPT4 用于模型的主观能力评估。

## 数据准备

我们提供了一个基于 [z-bench](https://github.com/zhenbench/z-bench) 的 demo 测试集：[subjective_demo.xlsx](https://opencompass.openxlab.space/utils/subjective_demo.xlsx)。

将主观问题集以.xlsx 格式存放在 `data/subjective/` 中。

表格包括以下字段：

- 'question'：问题描述
- 'index'：题目序号
- 'reference_answer'：参考答案
- 'evaluating_guidance'：评估引导
- 'capability'：题目所属的能力维度。

## 评测配置

具体流程包括:

1. 模型回答的推理
2. GPT4 评估比较对
3. 生成评测报告

对于 `config/subjective.py`，我们提供了部分注释，方便用户理解配置文件的含义。

```python
# 导入数据集与主观评测 summarizer
from mmengine.config import read_base
with read_base():
    from .datasets.subjective_cmp.subjective_cmp import subjective_datasets
    from .summarizers.subjective import summarizer

datasets = [*subjective_datasets]

from opencompass.models import HuggingFaceCausalLM, HuggingFace, OpenAI

#导入主观评测所需 partitioner 与 task
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask


# 定义推理和评测所需模型配置
# 包括推理模型 chatglm2-6b，qwen-7b-chat，internlm-chat-7b 和 评测模型 gpt4
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

# 定义主观评测配置
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        mode='all',  # 新参数，构建比较对时会交替构建两个
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2,  # 支持并行比较
        task=dict(
            type=SubjectiveEvalTask,  # 新 task，用来读入一对 model 的输入
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

## 启动评测

```shell
python run.py config/subjective.py -r
```

`-r` 参数支持复用模型推理和 GPT4 评估结果。

## 评测报告

评测报告会输出到 `output/.../summary/timestamp/report.md` ，包含胜率统计，对战分数与 ELO。具体格式如下：

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

对于评估模型 A 和 B 的比较对，有四种选择：

1. A 比 B 好
2. A 和 B 一样好
3. A 比 B 差
4. A 和 B 都不好

故 `win` / `tie` / `lose` / `not bad` 分别指模型 胜 / 平局 / 负 / 胜或一样好 的比例 。
`Bootstrap ELO` 是通过对比赛结果进行 1000 次随机顺序，计算出 ELO 分数的中位数。
