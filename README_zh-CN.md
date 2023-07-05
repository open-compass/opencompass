<div align="center">
  <!-- <img src="https://user-images.githubusercontent.com/22607038/250798681-b52045d2-cedd-4070-84e2-410903ac404f.png" width="500px"/> -->
  ![image](docs/zh_cn/_static/image/logo.svg)

[![docs](https://readthedocs.org/projects/opencompass/badge/?version=dev-1.x)](https://opencompass.readthedocs.io/en/dev-1.x/?badge=dev-1.x)
[![license](https://img.shields.io/github/license/IntenLM/opencompass.svg)](https://github.com/InternLM/opencompass/blob/main/LICENSE)
[![PyPI](https://badge.fury.io/py/opencompass.svg)](https://pypi.org/project/opencompass/)

[📘Documentation](https://opencompass.readthedocs.io/en/latest/) |
[🛠️Installation](https://opencompass.readthedocs.io/en/latest/get_started/install.html) |
[🤔Reporting Issues](https://github.com/InternLM/opencompass/issues/new/choose)

[English](/README.md) | 简体中文

</div>

## 介绍

OpenCompass 是面向大模型评测的一站式平台，旨在提供一套公平、公开、可复现的大模型评测基准方案。其主要特点如下：

- **模型及数据集的全方位支持**：预支持 20+ HuggingFace 及 API 模型，50+ 个数据集约 30 万题的的模型评测方案，五大维度全面评估模型能力。

- **高效分布式评测**：一行命令实现任务分割和分布式评测，数小时即可完成千亿模型全量评测\*。

- **多样化评测范式**：支持零样本、小样本及思维链评测，结合标准型或对话型提示词模板，轻松激发各种模型最大性能。

- **易于扩展的模块化设计**：想增加新模型或数据集？想要自定义更高级的任务分割策略，甚至接入新的集群管理系统？OpenCompass 的一切均可轻松扩展！

- **完善的实验记录及上报机制**：使用配置文件完整记录每一次实验，关键信息有迹可循；结果实时上报飞书机器人，第一时间知晓实验情况。

## 模型能力排名

## 数据集支持

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>语言</b>
      </td>
      <td>
        <b>知识</b>
      </td>
      <td>
        <b>推理</b>
      </td>
      <td>
        <b>考试</b>
      </td>
      <td>
        <b>理解</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
<details open>
<summary><b>字词释义</b></summary>

- WiC
- SummEdits

</details>

<details open>
<summary><b>成语习语</b></summary>

- CHID

</details>

<details open>
<summary><b>语义相似度</b></summary>

- AFQMC
- BUSTM

</details>

<details open>
<summary><b>指代消解</b></summary>

- CLUEWSC
- WSC
- WinoGrande

</details>

<details open>
<summary><b>翻译</b></summary>

- Flores

</details>
      </td>
      <td>
<details open>
<summary><b>知识问答</b></summary>

- BoolQ
- CommonSenseQA
- NaturalQuestion
- TrivialQA

</details>

<details open>
<summary><b>多语种问答</b></summary>

- TyDi-QA

</details>
      </td>
      <td>
<details open>
<summary><b>文本蕴含</b></summary>

- CMNLI
- OCNLI
- OCNLI_FC
- AX-b
- AX-g
- CB
- RTE

</details>

<details open>
<summary><b>常识推理</b></summary>

- StoryCloze
- StoryCloze-CN（即将上线）
- COPA
- ReCoRD
- HellaSwag
- PIQA
- SIQA

</details>

<details open>
<summary><b>数学推理</b></summary>

- MATH
- GSM8K

</details>

<details open>
<summary><b>定理应用</b></summary>

- TheoremQA

</details>

<details open>
<summary><b>代码</b></summary>

- HumanEval
- MBPP

</details>

<details open>
<summary><b>综合推理</b></summary>

- BBH

</details>
      </td>
      <td>
<details open>
<summary><b>初中、高中、大学、职业考试</b></summary>

- GAOKAO-2023
- CEval
- AGIEval
- MMLU
- GAOKAO-Bench
- MMLU-CN (即将上线)
- ARC

</details>
      </td>
      <td>
<details open>
<summary><b>阅读理解</b></summary>

- C3
- CMRC
- DRCD
- MultiRC
- RACE

</details>

<details open>
<summary><b>内容总结</b></summary>

- CSL
- LCSTS
- XSum

</details>

<details open>
<summary><b>内容分析</b></summary>

- EPRSTMT
- LAMBADA
- TNEWS

</details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

## 模型支持

<details close>
<summary><b>Huggingface 模型</b></summary>

- LLaMA
- Vicuna
- Alpaca
- Baichuan
- WizardLM
- ChatGLM-6B
- ChatGLM2-6B
- MPT
- Falcon
- TigerBot
- MOSS
- ……

</details>

<details close>
<summary><b>API 模型</b></summary>

- OpenAI
- Claude (即将推出)
- PaLM (即将推出)
- ……

</details>

<details close>
<summary><b>用户自定义模型</b></summary>

- GLM
- ……

</details>

## 安装

下面展示了快速安装的步骤。有部分第三方功能可能需要额外步骤才能正常运行，详细步骤请参考[安装指南](https://opencompass.readthedocs.io/zh_cn/latest/get_started.html)。

```Python
conda create --name opencompass python=3.8 pytorch torchvision -c pytorch -y
conda activate opencompass
git clone https://github.com/InternLM/opencompass opencompass
cd opencompass
pip install -e .
# 下载数据集到 data/ 处
# TODO: ....
```

## 评测

请阅读[快速上手](https://opencompass.readthedocs.io/zh_cn/latest/get_started.html)了解如何运行一个评测任务。

## 致谢

该项目部分的代码引用并修改自 [OpenICL](https://github.com/Shark-NLP/OpenICL)。

## 引用

```bibtex
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/InternLM/OpenCompass}},
    year={2023}
}
```
