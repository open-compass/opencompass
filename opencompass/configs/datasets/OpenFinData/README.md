# OpenFinData
## Introduction
The following introduction comes from the introduction in [OpenFinData](https://github.com/open-compass/OpenFinData)

```
OpenFinData是由东方财富与上海人工智能实验室联合发布的开源金融评测数据集。该数据集代表了最真实的产业场景需求，是目前场景最全、专业性最深的金融评测数据集。它基于东方财富实际金融业务的多样化丰富场景，旨在为金融科技领域的研究者和开发者提供一个高质量的数据资源。
OpenFinData is an open source financial evaluation dataset jointly released by Oriental Fortune and Shanghai Artificial Intelligence Laboratory. This data set represents the most realistic industrial scenario needs and is currently the most comprehensive and professional financial evaluation data set. It is based on the diverse and rich scenarios of Oriental Fortune's actual financial business and aims to provide a high-quality data resource for researchers and developers in the field of financial technology.
```

## Official link

### Repository

[OpenFinData](https://github.com/open-compass/OpenFinData)

## Use cases

In evaluation scripts, add OpenFinData dataset as other datasets by using
```
from .datasets.OepnFinData.OpenFinData_gen import OpenFinData_datasets
```

## Examples
Input example I:
```
你是一个数据审核小助手。表格内给出了2023年11月10日文一科技（600520）的最新数据，请指出其中哪个数据有误。请给出正确选项。
|     代码 | 名称   |    最新 |   涨幅% |   涨跌 |      成交量（股） |        成交额（元） |       流通市值 |        总市值 | 所属行业   |
|-------:|:-----|------:|------:|-----:|---------:|-----------:|-----------:|-----------:|:-------|
| 600520 | 文一科技 | 34.01 |  9.99 | 3.09 | 74227945 | 2472820896 | 5388200000 | 5388204300 | 通用设备   |
A. 2023年11月10日文一科技最新价34.01
B. 2023年11月10日文一科技成交额为2472820896
C. 文一科技的流通市值和总市值可能有误，因为流通市值5388200000元大于总市值5388204300元
D. 无明显错误数据
答案:
```
Output example I (from QWen-14B-Chat):
```
C. 文一科技的流通市值和总市值可能有误，因为流通市值5388200000元大于总市值5388204300元。
```
Input example II:
```
你是一个实体识别助手。请列出以下内容中提及的公司。
一度扬帆顺风的光伏产业，在过去几年中，面对潜在的高利润诱惑，吸引了众多非光伏行业的上市公司跨界转战，试图分得一杯羹。然而，今年下半年以来，出现了一个显著的趋势：一些跨界公司开始放弃或削减其光伏项目，包括皇氏集团（002329.SZ）、乐通股份（002319.SZ）、奥维通信（002231.SZ）等近十家公司。此外，还有一些光伏龙头放缓投资计划，如大全能源（688303.SH）、通威股份（600438.SZ）。业内人士表示，诸多因素导致了这股热潮的退却，包括市场变化、技术门槛、政策调整等等。光伏产业经历了从快速扩张到现在的理性回调，行业的自我调整和生态平衡正在逐步展现。从财务状况来看，较多选择退出的跨界企业都面临着经营压力。不过，皇氏集团、乐通股份等公司并未“全身而退”，仍在保持对光伏市场的关注，寻求进一步开拓的可能性。
答案:
```
Output example II (from InternLM2-7B-Chat):
```
皇氏集团（002329.SZ）、乐通股份（002319.SZ）、奥维通信（002231.SZ）、大全能源（688303.SH）、通威股份（600438.SZ）
```
## Evaluation results

```
dataset                             version    metric    mode      qwen-14b-chat-hf    internlm2-chat-7b-hf
----------------------------------  ---------  --------  ------  ------------------  ----------------------
OpenFinData-emotion_identification  b64193     accuracy  gen                  85.33                   78.67
OpenFinData-entity_disambiguation   b64193     accuracy  gen                  52                      68
OpenFinData-financial_facts         b64193     accuracy  gen                  70.67                   46.67
OpenFinData-data_inspection         a846b7     accuracy  gen                  53.33                   51.67
OpenFinData-financial_terminology   a846b7     accuracy  gen                  84                      73.33
OpenFinData-metric_calculation      a846b7     accuracy  gen                  55.71                   68.57
OpenFinData-value_extraction        a846b7     accuracy  gen                  84.29                   71.43
OpenFinData-intent_understanding    f0bd9e     accuracy  gen                  88                      86.67
OpenFinData-entity_recognition      81aeeb     accuracy  gen                  68                      84
```
