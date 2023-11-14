<div align="center">
  <img src="docs/zh_cn/_static/image/logo.svg" width="500px"/>
  <br />
  <br />

[![docs](https://readthedocs.org/projects/opencompass/badge)](https://opencompass.readthedocs.io/zh_CN)
[![license](https://img.shields.io/github/license/InternLM/opencompass.svg)](https://github.com/open-compass/opencompass/blob/main/LICENSE)

<!-- [![PyPI](https://badge.fury.io/py/opencompass.svg)](https://pypi.org/project/opencompass/) -->

[🌐Website](https://opencompass.org.cn/) |
[📘Documentation](https://opencompass.readthedocs.io/zh_CN/latest/index.html) |
[🛠️Installation](https://opencompass.readthedocs.io/zh_CN/latest/get_started/installation.html) |
[🤔Reporting Issues](https://github.com/open-compass/opencompass/issues/new/choose)

[English](/README.md) | 简体中文

</div>

<p align="center">
    👋 加入我们的 <a href="https://discord.gg/KKwfEbFj7U" target="_blank">Discord</a> 和 <a href="https://r.vansin.top/?r=opencompass" target="_blank">微信社区</a>
</p>

## 🧭	欢迎

来到**OpenCompass**！

就像指南针在我们的旅程中为我们导航一样，我们希望OpenCompass能够帮助你穿越评估大型语言模型的重重迷雾。OpenCompass提供丰富的算法和功能支持，期待OpenCompass能够帮助社区更便捷地对NLP模型的性能进行公平全面的评估。

🚩🚩🚩 欢迎加入 OpenCompass！我们目前**招聘全职研究人员/工程师和实习生**。如果您对 LLM 和 OpenCompass 充满热情，请随时通过[电子邮件](mailto:zhangsongyang@pjlab.org.cn)与我们联系。我们非常期待与您交流！

🔥🔥🔥 祝贺 **OpenCompass 作为大模型标准测试工具被Meta AI官方推荐**, 点击 Llama 的 [入门文档](https://ai.meta.com/llama/get-started/#validation) 获取更多信息.

> **注意**<br />
> 我们正式启动 OpenCompass 共建计划，诚邀社区用户为 OpenCompass 提供更具代表性和可信度的客观评测数据集!
> 点击 [Issue](https://github.com/open-compass/opencompass/issues/248) 获取更多数据集.
> 让我们携手共进，打造功能强大易用的大模型评测平台！

## 🚀 最新进展 <a><img width="35" height="20" src="https://user-images.githubusercontent.com/12782558/212848161-5e783dd6-11e8-4fe0-bbba-39ffb77730be.png"></a>

- **\[2023.11.13\]** 我们很高兴地宣布发布 OpenCompass v0.1.8 版本。此版本支持本地加载评估基准，从而无需连接互联网。请注意，随着此更新的发布，**您需要重新下载所有评估数据集**，以确保结果准确且最新。🔥🔥🔥。
- **\[2023.11.06\]** 我们已经支持了多个基于 API 的模型，包括ChatGLM Pro@智谱清言、ABAB-Chat@MiniMax 和讯飞。欢迎查看 [模型](https://opencompass.readthedocs.io/en/latest/user_guides/models.html) 部分以获取更多详细信息。🔥🔥🔥。
- **\[2023.10.24\]** 我们发布了一个全新的评测集，BotChat，用于评估大语言模型的多轮对话能力，欢迎查看 [BotChat](https://github.com/open-compass/BotChat) 获取更多信息. 🔥🔥🔥.
- **\[2023.09.26\]** 我们在评测榜单上更新了[Qwen](https://github.com/QwenLM/Qwen), 这是目前表现最好的开源模型之一, 欢迎访问[官方网站](https://opencompass.org.cn)获取详情.🔥🔥🔥.
- **\[2023.09.20\]** 我们在评测榜单上更新了[InternLM-20B](https://github.com/InternLM/InternLM), 欢迎访问[官方网站](https://opencompass.org.cn)获取详情.
- **\[2023.09.19\]** 我们在评测榜单上更新了WeMix-LLaMA2-70B/Phi-1.5-1.3B, 欢迎访问[官方网站](https://opencompass.org.cn)获取详情.
- **\[2023.09.18\]** 我们发布了[长文本评测指引](docs/zh_cn/advanced_guides/longeval.md).
- **\[2023.09.08\]** 我们在评测榜单上更新了Baichuan-2/Tigerbot-2/Vicuna-v1.5, 欢迎访问[官方网站](https://opencompass.org.cn)获取详情。
- **\[2023.09.06\]** 欢迎 [**Baichuan2**](https://github.com/baichuan-inc/Baichuan2) 团队采用OpenCompass对模型进行系统评估。我们非常感谢社区在提升LLM评估的透明度和可复现性上所做的努力。

> [更多](docs/zh_cn/notes/news.md)

## ✨ 介绍

![image](https://github.com/open-compass/opencompass/assets/22607038/30bcb2e2-3969-4ac5-9f29-ad3f4abb4f3b)

OpenCompass 是面向大模型评测的一站式平台。其主要特点如下：

- **开源可复现**：提供公平、公开、可复现的大模型评测方案

- **全面的能力维度**：五大维度设计，提供 70+ 个数据集约 40 万题的的模型评测方案，全面评估模型能力

- **丰富的模型支持**：已支持 20+ HuggingFace 及 API 模型

- **分布式高效评测**：一行命令实现任务分割和分布式评测，数小时即可完成千亿模型全量评测

- **多样化评测范式**：支持零样本、小样本及思维链评测，结合标准型或对话型提示词模板，轻松激发各种模型最大性能

- **灵活化拓展**：想增加新模型或数据集？想要自定义更高级的任务分割策略，甚至接入新的集群管理系统？OpenCompass 的一切均可轻松扩展！

## 📊 性能榜单

我们将陆续提供开源模型和API模型的具体性能榜单，请见 [OpenCompass Leaderbaord](https://opencompass.org.cn/rank) 。如需加入评测，请提供模型仓库地址或标准的 API 接口至邮箱  `opencompass@pjlab.org.cn`.

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 🛠️ 安装

下面展示了快速安装以及准备数据集的步骤。

```Python
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
# 下载数据集到 data/ 处
wget https://github.com/open-compass/opencompass/releases/download/0.1.8.rc1/OpenCompassData-core-20231110.zip
unzip OpenCompassData-core-20231110.zip
```

有部分第三方功能,如 Humaneval 以及 Llama,可能需要额外步骤才能正常运行，详细步骤请参考[安装指南](https://opencompass.readthedocs.io/zh_CN/latest/get_started/installation.html)。

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 🏗️ ️评测

确保按照上述步骤正确安装 OpenCompass 并准备好数据集后，可以通过以下命令评测 LLaMA-7b 模型在 MMLU 和 C-Eval 数据集上的性能：

```bash
python run.py --models hf_llama_7b --datasets mmlu_ppl ceval_ppl
```

OpenCompass 预定义了许多模型和数据集的配置，你可以通过 [工具](./docs/zh_cn/tools.md#ListConfigs) 列出所有可用的模型和数据集配置。

```bash
# 列出所有配置
python tools/list_configs.py
# 列出所有跟 llama 及 mmlu 相关的配置
python tools/list_configs.py llama mmlu
```

你也可以通过命令行去评测其它 HuggingFace 模型。同样以 LLaMA-7b 为例：

```bash
python run.py --datasets ceval_ppl mmlu_ppl \
--hf-path huggyllama/llama-7b \  # HuggingFace 模型地址
--model-kwargs device_map='auto' \  # 构造 model 的参数
--tokenizer-kwargs padding_side='left' truncation='left' use_fast=False \  # 构造 tokenizer 的参数
--max-out-len 100 \  # 最长生成 token 数
--max-seq-len 2048 \  # 模型能接受的最大序列长度
--batch-size 8 \  # 批次大小
--no-batch-padding \  # 不打开 batch padding，通过 for loop 推理，避免精度损失
--num-gpus 1  # 运行该模型所需的最少 gpu 数
```

> **注意**<br />
> 若需要运行上述命令，你需要删除所有从 `# ` 开始的注释。

通过命令行或配置文件，OpenCompass 还支持评测 API 或自定义模型，以及更多样化的评测策略。请阅读[快速开始](https://opencompass.readthedocs.io/zh_CN/latest/get_started/quick_start.html)了解如何运行一个评测任务。

更多教程请查看我们的[文档](https://opencompass.readthedocs.io/zh_CN/latest/index.html)。

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 📖 数据集支持

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
- IWSLT2017

</details>

<details open>
<summary><b>多语种问答</b></summary>

- TyDi-QA
- XCOPA

</details>

<details open>
<summary><b>多语种总结</b></summary>

- XLSum

</details>
      </td>
      <td>
<details open>
<summary><b>知识问答</b></summary>

- BoolQ
- CommonSenseQA
- NaturalQuestions
- TriviaQA

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
- ANLI

</details>

<details open>
<summary><b>常识推理</b></summary>

- StoryCloze
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
- StrategyQA
- SciBench

</details>

<details open>
<summary><b>综合推理</b></summary>

- BBH

</details>
      </td>
      <td>
<details open>
<summary><b>初中/高中/大学/职业考试</b></summary>

- C-Eval
- AGIEval
- MMLU
- GAOKAO-Bench
- CMMLU
- ARC
- Xiezhi

</details>

<details open>
<summary><b>医学考试</b></summary>

- CMB

</details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>理解</b>
      </td>
      <td>
        <b>长文本</b>
      </td>
      <td>
        <b>安全</b>
      </td>
      <td>
        <b>代码</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
<details open>
<summary><b>阅读理解</b></summary>

- C3
- CMRC
- DRCD
- MultiRC
- RACE
- DROP
- OpenBookQA
- SQuAD2.0

</details>

<details open>
<summary><b>内容总结</b></summary>

- CSL
- LCSTS
- XSum
- SummScreen

</details>

<details open>
<summary><b>内容分析</b></summary>

- EPRSTMT
- LAMBADA
- TNEWS

</details>
      </td>
      <td>
<details open>
<summary><b>长文本理解</b></summary>

- LEval
- LongBench
- GovReports
- NarrativeQA
- Qasper

</details>
      </td>
      <td>
<details open>
<summary><b>安全</b></summary>

- CivilComments
- CrowsPairs
- CValues
- JigsawMultilingual
- TruthfulQA

</details>
<details open>
<summary><b>健壮性</b></summary>

- AdvGLUE

</details>
      </td>
      <td>
<details open>
<summary><b>代码</b></summary>

- HumanEval
- HumanEvalX
- MBPP
- APPs
- DS1000

</details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 📖 模型支持

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

- InternLM
- LLaMA
- Vicuna
- Alpaca
- Baichuan
- WizardLM
- ChatGLM2
- Falcon
- TigerBot
- Qwen
- ……

</td>
<td>

- OpenAI
- Claude
- PaLM (即将推出)
- ……

</td>

</tr>
  </tbody>
</table>

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 🔜 路线图

- [ ] 主观评测
  - [ ] 发布主观评测榜单
  - [ ] 发布主观评测数据集
- [ ] 长文本
  - [ ] 支持广泛的长文本评测集
  - [ ] 发布长文本评测榜单
- [ ] 代码能力
  - [ ] 发布代码能力评测榜单
  - [ ] 提供非Python语言的评测服务
- [ ] 智能体
  - [ ] 支持丰富的智能体方案
  - [ ] 提供智能体评测榜单
- [ ] 鲁棒性
  - [ ] 支持各类攻击方法

## 👷‍♂️ 贡献

我们感谢所有的贡献者为改进和提升 OpenCompass 所作出的努力。请参考[贡献指南](https://opencompass.readthedocs.io/zh_CN/latest/notes/contribution_guide.html)来了解参与项目贡献的相关指引。

## 🤝 致谢

该项目部分的代码引用并修改自 [OpenICL](https://github.com/Shark-NLP/OpenICL)。

该项目部分的数据集和提示词实现修改自 [chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub), [instruct-eval](https://github.com/declare-lab/instruct-eval)

## 🖊️ 引用

```bibtex
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}
```

<p align="right"><a href="#top">🔝返回顶部</a></p>
