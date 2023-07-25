<div align="center">
  <img src="docs/zh_cn/_static/image/logo.svg" width="500px"/>
  <br />
  <br />

[![docs](https://readthedocs.org/projects/opencompass/badge)](https://opencompass.readthedocs.io/zh_CN)
[![license](https://img.shields.io/github/license/InternLM/opencompass.svg)](https://github.com/InternLM/opencompass/blob/main/LICENSE)

<!-- [![PyPI](https://badge.fury.io/py/opencompass.svg)](https://pypi.org/project/opencompass/) -->

[🌐Website](https://opencompass.org.cn/) |
[📘Documentation](https://opencompass.readthedocs.io/zh_CN/latest/index.html) |
[🛠️Installation](https://opencompass.readthedocs.io/zh_CN/latest/get_started.html) |
[🤔Reporting Issues](https://github.com/InternLM/opencompass/issues/new/choose)

[English](/README.md) | 简体中文

</div>

<p align="center">
    👋 加入我们的 <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> 和 <a href="https://github.com/InternLM/InternLM/assets/25839884/a6aad896-7232-4220-ac84-9e070c2633ce" target="_blank">微信社区</a>
</p>

欢迎来到OpenCompass！

就像指南针在我们的旅程中为我们导航一样，我们希望OpenCompass能够帮助你穿越评估大型语言模型的重重迷雾。OpenCompass提供丰富的算法和功能支持，期待OpenCompass能够帮助社区更便捷地对NLP模型的性能进行公平全面的评估。

## 更新

- **\[2023.07.19\]** 新增了 [Llama 2](https://ai.meta.com/llama/)！我们近期将会公布其评测结果。\[[文档](./docs/zh_cn/get_started.md#安装)\]
- **\[2023.07.13\]** 发布了 [MMBench](https://opencompass.org.cn/MMBench)，该数据集经过细致整理，用于评测多模态模型全方位能力 🔥🔥🔥。

## 介绍

OpenCompass 是面向大模型评测的一站式平台。其主要特点如下：

- **开源可复现**：提供公平、公开、可复现的大模型评测方案

- **全面的能力维度**：五大维度设计，提供 50+ 个数据集约 30 万题的的模型评测方案，全面评估模型能力

- **丰富的模型支持**：已支持 20+ HuggingFace 及 API 模型

- **分布式高效评测**：一行命令实现任务分割和分布式评测，数小时即可完成千亿模型全量评测

- **多样化评测范式**：支持零样本、小样本及思维链评测，结合标准型或对话型提示词模板，轻松激发各种模型最大性能

- **灵活化拓展**：想增加新模型或数据集？想要自定义更高级的任务分割策略，甚至接入新的集群管理系统？OpenCompass 的一切均可轻松扩展！

## 性能榜单

我们将陆续提供开源模型和API模型的具体性能榜单，请见 [OpenCompass Leaderbaord](https://opencompass.org.cn/rank) 。如需加入评测，请提供模型仓库地址或标准的 API 接口至邮箱  `opencompass@pjlab.org.cn`.

[![image](https://github.com/InternLM/opencompass/assets/13503330/76237116-a9dd-4207-abef-7ff73b89568a)](https://opencompass.org.cn/rank)

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
        <b>学科</b>
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
<summary><b>初中/高中/大学/职业考试</b></summary>

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

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>开源模型</b>
      </td>
      <td>
        <b>API 模型</b>
      </td>
      <!-- <td>
        <b>自定义模型</b>
      </td> -->
    </tr>
    <tr valign="top">
      <td>

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

</td>
<td>

- OpenAI
- Claude (即将推出)
- PaLM (即将推出)
- ……

</td>
<!-- <td>

- GLM
- ……

</td> -->
</tr>
  </tbody>
</table>

## 安装

下面展示了快速安装的步骤。有部分第三方功能可能需要额外步骤才能正常运行，详细步骤请参考[安装指南](https://opencompass.readthedocs.io/zh_cn/latest/get_started.html)。

```Python
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/InternLM/opencompass opencompass
cd opencompass
pip install -e .
# 下载数据集到 data/ 处
wget https://github.com/InternLM/opencompass/releases/download/0.1.0/OpenCompassData.zip
unzip OpenCompassData.zip
```

## 评测

请阅读[快速上手](https://opencompass.readthedocs.io/zh_CN/latest/get_started.html#id2)了解如何运行一个评测任务。

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
