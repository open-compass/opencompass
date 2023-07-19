<div align="center">
  <img src="docs/en/_static/image/logo.svg" width="500px"/>
  <br />
  <br />

[![docs](https://readthedocs.org/projects/opencompass/badge)](https://opencompass.readthedocs.io/en)
[![license](https://img.shields.io/github/license/InternLM/opencompass.svg)](https://github.com/InternLM/opencompass/blob/main/LICENSE)

<!-- [![PyPI](https://badge.fury.io/py/opencompass.svg)](https://pypi.org/project/opencompass/) -->

[🌐Website](https://opencompass.org.cn/) |
[📘Documentation](https://opencompass.readthedocs.io/en/latest/) |
[🛠️Installation](https://opencompass.readthedocs.io/en/latest/get_started/install.html) |
[🤔Reporting Issues](https://github.com/InternLM/opencompass/issues/new/choose)

English | [简体中文](README_zh-CN.md)

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://github.com/InternLM/InternLM/assets/25839884/a6aad896-7232-4220-ac84-9e070c2633ce" target="_blank">WeChat</a>
</p>

Welcome to **OpenCompass**!

Just like a compass guides us on our journey, OpenCompass will guide you through the complex landscape of evaluating large language models. With its powerful algorithms and intuitive interface, OpenCompass makes it easy to assess the quality and effectiveness of your NLP models.

## News

- **\[2023.07.19\]** We have supported [Llama 2](https://ai.meta.com/llama/)! Its performance report will be available soon. \[[doc](./docs/en/get_started.md#Installation)\]
- **\[2023.07.13\]** We release [MMBench](https://opencompass.org.cn/MMBench), a meticulously curated dataset to comprehensively evaluate different abilities of multimodality models 🔥🔥🔥.

## Introduction

OpenCompass is a one-stop platform for large model evaluation, aiming to provide a fair, open, and reproducible benchmark for large model evaluation. Its main features includes:

- **Comprehensive support for models and datasets**: Pre-support for 20+ HuggingFace and API models, a model evaluation scheme of 50+ datasets with about 300,000 questions, comprehensively evaluating the capabilities of the models in five dimensions.

- **Efficient distributed evaluation**: One line command to implement task division and distributed evaluation, completing the full evaluation of billion-scale models in just a few hours.

- **Diversified evaluation paradigms**: Support for zero-shot, few-shot, and chain-of-thought evaluations, combined with standard or dialogue type prompt templates, to easily stimulate the maximum performance of various models.

- **Modular design with high extensibility**: Want to add new models or datasets, customize an advanced task division strategy, or even support a new cluster management system? Everything about OpenCompass can be easily expanded!

- **Experiment management and reporting mechanism**: Use config files to fully record each experiment, support real-time reporting of results.

## Leaderboard

We provide [OpenCompass Leaderbaord](https://opencompass.org.cn/rank) for community to rank all public models and API models. If you would like to join the evaluation, please provide the model repository URL or a standard API interface to the email address `opencompass@pjlab.org.cn`.

[![image](https://github.com/InternLM/opencompass/assets/13503330/80c5a42c-ddf0-4c6f-b39e-c175711ac381)](https://opencompass.org.cn/rank)

## Dataset Support

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Language</b>
      </td>
      <td>
        <b>Knowledge</b>
      </td>
      <td>
        <b>Reasoning</b>
      </td>
      <td>
        <b>Comprehensive Examination</b>
      </td>
      <td>
        <b>Understanding</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
<details open>
<summary><b>Word Definition</b></summary>

- WiC
- SummEdits

</details>

<details open>
<summary><b>Idiom Learning</b></summary>

- CHID

</details>

<details open>
<summary><b>Semantic Similarity</b></summary>

- AFQMC
- BUSTM

</details>

<details open>
<summary><b>Coreference Resolution</b></summary>

- CLUEWSC
- WSC
- WinoGrande

</details>

<details open>
<summary><b>Translation</b></summary>

- Flores

</details>
      </td>
      <td>
<details open>
<summary><b>Knowledge Question Answering</b></summary>

- BoolQ
- CommonSenseQA
- NaturalQuestion
- TrivialQA

</details>

<details open>
<summary><b>Multi-language Question Answering</b></summary>

- TyDi-QA

</details>
      </td>
      <td>
<details open>
<summary><b>Textual Entailment</b></summary>

- CMNLI
- OCNLI
- OCNLI_FC
- AX-b
- AX-g
- CB
- RTE

</details>

<details open>
<summary><b>Commonsense Reasoning</b></summary>

- StoryCloze
- StoryCloze-CN (coming soon)
- COPA
- ReCoRD
- HellaSwag
- PIQA
- SIQA

</details>

<details open>
<summary><b>Mathematical Reasoning</b></summary>

- MATH
- GSM8K

</details>

<details open>
<summary><b>Theorem Application</b></summary>

- TheoremQA

</details>

<details open>
<summary><b>Code</b></summary>

- HumanEval
- MBPP

</details>

<details open>
<summary><b>Comprehensive Reasoning</b></summary>

- BBH

</details>
      </td>
      <td>
<details open>
<summary><b>Junior High, High School, University, Professional Examinations</b></summary>

- GAOKAO-2023
- CEval
- AGIEval
- MMLU
- GAOKAO-Bench
- MMLU-CN (coming soon)
- ARC

</details>
      </td>
      <td>
<details open>
<summary><b>Reading Comprehension</b></summary>

- C3
- CMRC
- DRCD
- MultiRC
- RACE

</details>

<details open>
<summary><b>Content Summary</b></summary>

- CSL
- LCSTS
- XSum

</details>

<details open>
<summary><b>Content Analysis</b></summary>

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

## Model Support

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Open-source Models</b>
      </td>
      <td>
        <b>API Models</b>
      </td>
      <!-- <td>
        <b>Custom Models</b>
      </td> -->
    </tr>
    <tr valign="top">
      <td>

- InternLM
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
- ...

</td>
<td>

- OpenAI
- Claude (coming soon)
- PaLM (coming soon)
- ……

</td>

<!--
- GLM
- ...

</td> -->

</tr>
  </tbody>
</table>

## Installation

Below are the steps for quick installation. Some third-party features may require additional steps to work properly, for detailed steps please refer to the [Installation Guide](https://opencompass.readthedocs.io/en/latest/get_started.html).

```Python
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/InternLM/opencompass opencompass
cd opencompass
pip install -e .
# Download dataset to data/ folder
wget https://github.com/InternLM/opencompass/releases/download/0.1.0/OpenCompassData.zip
unzip OpenCompassData.zip
```

## Evaluation

Please read the [Quick Start](https://opencompass.readthedocs.io/en/latest/get_started.html) to learn how to run an evaluation task.

## Acknowledgements

Some code in this project is cited and modified from [OpenICL](https://github.com/Shark-NLP/OpenICL).

## Citation

```bibtex
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/InternLM/OpenCompass}},
    year={2023}
}
```
