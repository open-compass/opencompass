<div align="center">
  <img src="docs/zh_cn/_static/image/logo.svg" width="500px"/>
  <br />
  <br />

[![][github-release-shield]][github-release-link]
[![][github-releasedate-shield]][github-releasedate-link]
[![][github-contributors-shield]][github-contributors-link]<br>
[![][github-forks-shield]][github-forks-link]
[![][github-stars-shield]][github-stars-link]
[![][github-issues-shield]][github-issues-link]
[![][github-license-shield]][github-license-link]

<!-- [![PyPI](https://badge.fury.io/py/opencompass.svg)](https://pypi.org/project/opencompass/) -->

[🌐官方网站](https://opencompass.org.cn/) |
[📖数据集社区](https://hub.opencompass.org.cn/home) |
[📊性能榜单](https://rank.opencompass.org.cn/home) |
[📘文档教程](https://opencompass.readthedocs.io/zh_CN/latest/index.html) |
[🛠️安装](https://opencompass.readthedocs.io/zh_CN/latest/get_started/installation.html) |
[🤔报告问题](https://github.com/open-compass/opencompass/issues/new/choose)

[English](/README.md) | 简体中文

[![][github-trending-shield]][github-trending-url]

</div>

<p align="center">
    👋 加入我们的 <a href="https://discord.gg/KKwfEbFj7U" target="_blank">Discord</a> 和 <a href="https://r.vansin.top/?r=opencompass" target="_blank">微信社区</a>
</p>

> \[!IMPORTANT\]
>
> **收藏项目**，你将能第一时间获取 OpenCompass 的最新动态～⭐️

<details>
  <summary><kbd>Star History</kbd></summary>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=open-compass%2Fopencompass&theme=dark&type=Date">
    <img width="100%" src="https://api.star-history.com/svg?repos=open-compass%2Fopencompass&type=Date">
  </picture>
</details>

## 🧭	欢迎

来到**OpenCompass**！

就像指南针在我们的旅程中为我们导航一样，我们希望OpenCompass能够帮助你穿越评估大型语言模型的重重迷雾。OpenCompass提供丰富的算法和功能支持，期待OpenCompass能够帮助社区更便捷地对NLP模型的性能进行公平全面的评估。

🚩🚩🚩 欢迎加入 OpenCompass！我们目前**招聘全职研究人员/工程师和实习生**。如果您对 LLM 和 OpenCompass 充满热情，请随时通过[电子邮件](mailto:zhangsongyang@pjlab.org.cn)与我们联系。我们非常期待与您交流！

🔥🔥🔥 祝贺 **OpenCompass 作为大模型标准测试工具被Meta AI官方推荐**, 点击 Llama 的 [入门文档](https://ai.meta.com/llama/get-started/#validation) 获取更多信息。

> **注意**<br />
> 重要通知：从 v0.4.0 版本开始，所有位于 ./configs/datasets、./configs/models 和 ./configs/summarizers 目录下的 AMOTIC 配置文件将迁移至 opencompass 包中。请及时更新您的配置文件路径。

## 🚀 最新进展 <a><img width="35" height="20" src="https://user-images.githubusercontent.com/12782558/212848161-5e783dd6-11e8-4fe0-bbba-39ffb77730be.png"></a>

- **\[2025.04.01\]** OpenCompass 现已支持 `CascadeEvaluator`，允许多个评估器按顺序工作，可以为更复杂的评估场景创建自定义评估流程，查看[文档](docs/zh_cn/advanced_guides/llm_judge.md)了解具体用法！🔥🔥🔥
- **\[2025.03.11\]** 现已支持 `SuperGPQA`  覆盖285 个研究生学科的知识能力评测，欢迎尝试！🔥🔥🔥
- **\[2025.02.28\]** 我们为 `DeepSeek-R1` 系列模型添加了教程，请查看 [评估推理模型](docs/zh_cn/user_guides/deepseek_r1.md) 了解更多详情！🔥🔥🔥
- **\[2025.02.15\]** 我们新增了两个实用的评测工具：用于LLM作为评判器的`GenericLLMEvaluator`和用于数学推理评估的`MATHEvaluator`。查看[LLM评判器](docs/zh_cn/advanced_guides/llm_judge.md)和[数学能力评测](docs/zh_cn/advanced_guides/general_math.md)文档了解更多详情！🔥🔥🔥
- **\[2025.01.16\]** 我们现已支持 [InternLM3-8B-Instruct](https://huggingface.co/internlm/internlm3-8b-instruct) 模型，该模型在推理、知识类任务上取得同量级最优性能，欢迎尝试。
- **\[2024.12.17\]** 我们提供了12月CompassAcademic学术榜单评估脚本 [CompassAcademic](configs/eval_academic_leaderboard_202412.py)，你可以通过简单地配置复现官方评测结果。
- **\[2024.10.14\]** 现已支持OpenAI多语言问答数据集[MMMLU](https://huggingface.co/datasets/openai/MMMLU)，欢迎尝试! 🔥🔥🔥
- **\[2024.09.19\]** 现已支持[Qwen2.5](https://huggingface.co/Qwen)(0.5B to 72B) ，可以使用多种推理后端(huggingface/vllm/lmdeploy), 欢迎尝试! 🔥🔥🔥
- **\[2024.09.05\]** 现已支持OpenAI o1 模型(`o1-mini-2024-09-12` and `o1-preview-2024-09-12`), 欢迎尝试! 🔥🔥🔥
- **\[2024.09.05\]** OpenCompass 现在支持通过模型后处理来进行答案提取，以更准确地展示模型的能力。作为此次更新的一部分，我们集成了 [XFinder](https://github.com/IAAR-Shanghai/xFinder) 作为首个后处理模型。具体信息请参阅 [文档](opencompass/utils/postprocessors/xfinder/README.md)，欢迎尝试！ 🔥🔥🔥
- **\[2024.08.20\]** OpenCompass 现已支持 [SciCode](https://github.com/scicode-bench/SciCode): A Research Coding Benchmark Curated by Scientists。 🔥🔥🔥
- **\[2024.08.16\]** OpenCompass 现已支持全新的长上下文语言模型评估基准——[RULER](https://arxiv.org/pdf/2404.06654)。RULER 通过灵活的配置，提供了对长上下文包括检索、多跳追踪、聚合和问答等多种任务类型的评测，欢迎访问[RULER](configs/datasets/ruler/README.md)。🔥🔥🔥
- **\[2024.07.23\]** 我们支持了[Gemma2](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)模型，欢迎试用！🔥🔥🔥
- **\[2024.07.23\]** 我们支持了[ModelScope](www.modelscope.cn)数据集，您可以按需加载，无需事先下载全部数据到本地，欢迎试用！🔥🔥🔥
- **\[2024.07.17\]** 我们发布了CompassBench-202407榜单的示例数据和评测规则，敬请访问 [CompassBench](https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/compassbench_intro.html) 获取更多信息。 🔥🔥🔥
- **\[2024.07.17\]** 我们正式发布 NeedleBench 的[技术报告](http://arxiv.org/abs/2407.11963)。诚邀您访问我们的[帮助文档](https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/needleinahaystack_eval.html)进行评估。🔥🔥🔥
- **\[2024.07.04\]** OpenCompass 现已支持 InternLM2.5， 它拥有卓越的推理性能、有效支持百万字超长上下文以及工具调用能力整体升级，欢迎访问[OpenCompass Config](https://github.com/open-compass/opencompass/tree/main/configs/models/hf_internlm) 和 [InternLM](https://github.com/InternLM/InternLM) .🔥🔥🔥.
- **\[2024.06.20\]** OpenCompass 现已支持一键切换推理加速后端，助力评测过程更加高效。除了默认的HuggingFace推理后端外，还支持了常用的 [LMDeploy](https://github.com/InternLM/lmdeploy) 和 [vLLM](https://github.com/vllm-project/vllm) ，支持命令行一键切换和部署 API 加速服务两种方式，详细使用方法见[文档](docs/zh_cn/advanced_guides/accelerator_intro.md)。欢迎试用！🔥🔥🔥.

> [更多](docs/zh_cn/notes/news.md)

## 📊 性能榜单

我们将陆续提供开源模型和 API 模型的具体性能榜单，请见 [OpenCompass Leaderboard](https://rank.opencompass.org.cn/home) 。如需加入评测，请提供模型仓库地址或标准的 API 接口至邮箱  `opencompass@pjlab.org.cn`.

你也可以参考[CompassAcademic](configs/eval_academic_leaderboard_202412.py)，快速地复现榜单的结果，目前选取的数据集包括 综合知识推理 (MMLU-Pro/GPQA Diamond) ,逻辑推理 (BBH) ,数学推理 (MATH-500, AIME) ,代码生成 (LiveCodeBench, HumanEval) ,指令跟随 (IFEval) 。

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 🛠️ 安装指南

下面提供了快速安装和数据集准备的步骤。

### 💻 环境搭建

我们强烈建议使用 `conda` 来管理您的 Python 环境。

- #### 创建虚拟环境

  ```bash
  conda create --name opencompass python=3.10 -y
  conda activate opencompass
  ```

- #### 通过pip安装OpenCompass

  ```bash
  # 支持绝大多数数据集及模型
  pip install -U opencompass

  # 完整安装（支持更多数据集）
  # pip install "opencompass[full]"

  # 模型推理后端，由于这些推理后端通常存在依赖冲突，建议使用不同的虚拟环境来管理它们。
  # pip install "opencompass[lmdeploy]"
  # pip install "opencompass[vllm]"

  # API 测试（例如 OpenAI、Qwen）
  # pip install "opencompass[api]"
  ```

- #### 基于源码安装OpenCompass

  如果希望使用 OpenCompass 的最新功能，也可以从源代码构建它：

  ```bash
  git clone https://github.com/open-compass/opencompass opencompass
  cd opencompass
  pip install -e .
  # pip install -e ".[full]"
  # pip install -e ".[vllm]"
  ```

### 📂 数据准备

#### 提前离线下载

OpenCompass支持使用本地数据集进行评测，数据集的下载和解压可以通过以下命令完成：

```bash
# 下载数据集到 data/ 处
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

#### 从 OpenCompass 自动下载

我们已经支持从OpenCompass存储服务器自动下载数据集。您可以通过额外的 `--dry-run` 参数来运行评估以下载这些数据集。
目前支持的数据集列表在[这里](https://github.com/open-compass/opencompass/blob/main/opencompass/utils/datasets_info.py#L259)。更多数据集将会很快上传。

#### (可选) 使用 ModelScope 自动下载

另外，您还可以使用[ModelScope](www.modelscope.cn)来加载数据集：
环境准备：

```bash
pip install modelscope
export DATASET_SOURCE=ModelScope
```

配置好环境后，无需下载全部数据，直接提交评测任务即可。目前支持的数据集有：

```bash
humaneval, triviaqa, commonsenseqa, tydiqa, strategyqa, cmmlu, lambada, piqa, ceval, math, LCSTS, Xsum, winogrande, openbookqa, AGIEval, gsm8k, nq, race, siqa, mbpp, mmlu, hellaswag, ARC, BBH, xstory_cloze, summedits, GAOKAO-BENCH, OCNLI, cmnli
```

有部分第三方功能,如 Humaneval 以及 Llama,可能需要额外步骤才能正常运行，详细步骤请参考[安装指南](https://opencompass.readthedocs.io/zh_CN/latest/get_started/installation.html)。

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 🏗️ ️评测

在确保按照上述步骤正确安装了 OpenCompass 并准备好了数据集之后，现在您可以开始使用 OpenCompass 进行首次评估！

- ### 首次评测

  OpenCompass 支持通过命令行界面 (CLI) 或 Python 脚本来设置配置。对于简单的评估设置，我们推荐使用 CLI；而对于更复杂的评估，则建议使用脚本方式。你可以在examples文件夹下找到更多脚本示例。

  ```bash
  # 命令行界面 (CLI)
  opencompass --models hf_internlm2_5_1_8b_chat --datasets demo_gsm8k_chat_gen

  # Python 脚本
  opencompass examples/eval_chat_demo.py
  ```

  你可以在[examples](./examples) 文件夹下找到更多的脚本示例。

- ### API评测

  OpenCompass 在设计上并不区分开源模型与 API 模型。您可以以相同的方式或甚至在同一设置中评估这两种类型的模型。

  ```bash
  export OPENAI_API_KEY="YOUR_OPEN_API_KEY"
  # 命令行界面 (CLI)
  opencompass --models gpt_4o_2024_05_13 --datasets demo_gsm8k_chat_gen

  # Python 脚本
  opencompass  examples/eval_api_demo.py


  # 现已支持 o1_mini_2024_09_12/o1_preview_2024_09_12  模型, 默认情况下 max_completion_tokens=8192.
  ```

- ### 推理后端

  另外，如果您想使用除 HuggingFace 之外的推理后端来进行加速评估，比如 LMDeploy 或 vLLM，可以通过以下命令进行。请确保您已经为所选的后端安装了必要的软件包，并且您的模型支持该后端的加速推理。更多信息，请参阅关于推理加速后端的文档 [这里](docs/zh_cn/advanced_guides/accelerator_intro.md)。以下是使用 LMDeploy 的示例：

  ```bash
  opencompass --models hf_internlm2_5_1_8b_chat --datasets demo_gsm8k_chat_gen -a lmdeploy
  ```

- ### 支持的模型与数据集

  OpenCompass 预定义了许多模型和数据集的配置，你可以通过 [工具](./docs/zh_cn/tools.md#ListConfigs) 列出所有可用的模型和数据集配置。

  ```bash
  # 列出所有配置
  python tools/list_configs.py
  # 列出所有跟 llama 及 mmlu 相关的配置
  python tools/list_configs.py llama mmlu
  ```

  #### 支持的模型

  如果模型不在列表中，但支持 Huggingface AutoModel 类或支持针对 OpenAI 接口的推理引擎封装（详见[官方文档](https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/new_model.html)），您仍然可以使用 OpenCompass 对其进行评估。欢迎您贡献维护 OpenCompass 支持的模型和数据集列表。

  ```bash
  opencompass --datasets demo_gsm8k_chat_gen --hf-type chat --hf-path internlm/internlm2_5-1_8b-chat
  ```

  #### 支持的数据集

  目前，OpenCompass针对数据集给出了标准的推荐配置。通常，`_gen.py`或`_llm_judge_gen.py`为结尾的配置文件将指向我们为该数据集提供的推荐配置。您可以参阅[官方文档](https://opencompass.readthedocs.io/zh-cn/latest/dataset_statistics.html) 的数据集统计章节来获取详细信息。

  ```bash
  # 基于规则的推荐配置
  opencompass --datasets aime2024_gen --models hf_internlm2_5_1_8b_chat

  # 基于LLM Judge的推荐配置
  opencompass --datasets aime2024_llm_judge_gen --models hf_internlm2_5_1_8b_chat
  ```

  此外，如果你想在多块 GPU 上使用模型进行推理，您可以使用 `--max-num-worker` 参数。

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 opencompass --datasets demo_gsm8k_chat_gen --hf-type chat --hf-path internlm/internlm2_5-1_8b-chat --max-num-worker 2
  ```

> \[!TIP\]
>
> `--hf-num-gpus` 用于 模型并行(huggingface 格式)，`--max-num-worker` 用于数据并行。

> \[!TIP\]
>
> configuration with `_ppl` is designed for base model typically.
> 配置带 `_ppl` 的配置设计给基础模型使用。
> 配置带 `_gen` 的配置可以同时用于基础模型和对话模型。

通过命令行或配置文件，OpenCompass 还支持评测 API 或自定义模型，以及更多样化的评测策略。请阅读[快速开始](https://opencompass.readthedocs.io/zh_CN/latest/get_started/quick_start.html)了解如何运行一个评测任务。

更多教程请查看我们的[文档](https://opencompass.readthedocs.io/zh_CN/latest/index.html)。

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 📣 OpenCompass 2.0

我们很高兴发布 OpenCompass 司南 2.0 大模型评测体系，它主要由三大核心模块构建而成：[CompassKit](https://github.com/open-compass)、[CompassHub](https://hub.opencompass.org.cn/home)以及[CompassRank](https://rank.opencompass.org.cn/home)。

**CompassRank** 系统进行了重大革新与提升，现已成为一个兼容并蓄的排行榜体系，不仅囊括了开源基准测试项目，还包含了私有基准测试。此番升级极大地拓宽了对行业内各类模型进行全面而深入测评的可能性。

**CompassHub** 创新性地推出了一个基准测试资源导航平台，其设计初衷旨在简化和加快研究人员及行业从业者在多样化的基准测试库中进行搜索与利用的过程。为了让更多独具特色的基准测试成果得以在业内广泛传播和应用，我们热忱欢迎各位将自定义的基准数据贡献至CompassHub平台。只需轻点鼠标，通过访问[这里](https://hub.opencompass.org.cn/dataset-submit)，即可启动提交流程。

**CompassKit** 是一系列专为大型语言模型和大型视觉-语言模型打造的强大评估工具合集，它所提供的全面评测工具集能够有效地对这些复杂模型的功能性能进行精准测量和科学评估。在此，我们诚挚邀请您在学术研究或产品研发过程中积极尝试运用我们的工具包，以助您取得更加丰硕的研究成果和产品优化效果。

## ✨ 介绍

![image](https://github.com/open-compass/opencompass/assets/22607038/30bcb2e2-3969-4ac5-9f29-ad3f4abb4f3b)

OpenCompass 是面向大模型评测的一站式平台。其主要特点如下：

- **开源可复现**：提供公平、公开、可复现的大模型评测方案

- **全面的能力维度**：五大维度设计，提供 70+ 个数据集约 40 万题的的模型评测方案，全面评估模型能力

- **丰富的模型支持**：已支持 20+ HuggingFace 及 API 模型

- **分布式高效评测**：一行命令实现任务分割和分布式评测，数小时即可完成千亿模型全量评测

- **多样化评测范式**：支持零样本、小样本及思维链评测，结合标准型或对话型提示词模板，轻松激发各种模型最大性能

- **灵活化拓展**：想增加新模型或数据集？想要自定义更高级的任务分割策略，甚至接入新的集群管理系统？OpenCompass 的一切均可轻松扩展！

## 📖 数据集支持

我们已经在OpenCompass官网的文档中支持了所有可在本平台上使用的数据集的统计列表。

您可以通过排序、筛选和搜索等功能从列表中快速找到您需要的数据集。

详情请参阅 [官方文档](https://opencompass.readthedocs.io/zh-cn/latest/dataset_statistics.html) 的数据集统计章节。

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

- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Baichuan](https://github.com/baichuan-inc)
- [BlueLM](https://github.com/vivo-ai-lab/BlueLM)
- [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)
- [ChatGLM3](https://github.com/THUDM/ChatGLM3-6B)
- [Gemma](https://huggingface.co/google/gemma-7b)
- [InternLM](https://github.com/InternLM/InternLM)
- [LLaMA](https://github.com/facebookresearch/llama)
- [LLaMA3](https://github.com/meta-llama/llama3)
- [Qwen](https://github.com/QwenLM/Qwen)
- [TigerBot](https://github.com/TigerResearch/TigerBot)
- [Vicuna](https://github.com/lm-sys/FastChat)
- [WizardLM](https://github.com/nlpxucan/WizardLM)
- [Yi](https://github.com/01-ai/Yi)
- ……

</td>
<td>

- OpenAI
- Gemini
- Claude
- ZhipuAI(ChatGLM)
- Baichuan
- ByteDance(YunQue)
- Huawei(PanGu)
- 360
- Baidu(ERNIEBot)
- MiniMax(ABAB-Chat)
- SenseTime(nova)
- Xunfei(Spark)
- ……

</td>

</tr>
  </tbody>
</table>

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 🔜 路线图

- [x] 主观评测
  - [x] 发布主观评测榜单
  - [x] 发布主观评测数据集
- [x] 长文本
  - [x] 支持广泛的长文本评测集
  - [ ] 发布长文本评测榜单
- [x] 代码能力
  - [ ] 发布代码能力评测榜单
  - [x] 提供非Python语言的评测服务
- [x] 智能体
  - [ ] 支持丰富的智能体方案
  - [x] 提供智能体评测榜单
- [x] 鲁棒性
  - [x] 支持各类攻击方法

## 👷‍♂️ 贡献

我们感谢所有的贡献者为改进和提升 OpenCompass 所作出的努力。请参考[贡献指南](https://opencompass.readthedocs.io/zh_CN/latest/notes/contribution_guide.html)来了解参与项目贡献的相关指引。

<a href="https://github.com/open-compass/opencompass/graphs/contributors" target="_blank">
  <table>
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=open-compass/opencompass"><br><br>
      </th>
    </tr>
  </table>
</a>

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

[github-contributors-link]: https://github.com/open-compass/opencompass/graphs/contributors
[github-contributors-shield]: https://img.shields.io/github/contributors/open-compass/opencompass?color=c4f042&labelColor=black&style=flat-square
[github-forks-link]: https://github.com/open-compass/opencompass/network/members
[github-forks-shield]: https://img.shields.io/github/forks/open-compass/opencompass?color=8ae8ff&labelColor=black&style=flat-square
[github-issues-link]: https://github.com/open-compass/opencompass/issues
[github-issues-shield]: https://img.shields.io/github/issues/open-compass/opencompass?color=ff80eb&labelColor=black&style=flat-square
[github-license-link]: https://github.com/open-compass/opencompass/blob/main/LICENSE
[github-license-shield]: https://img.shields.io/github/license/open-compass/opencompass?color=white&labelColor=black&style=flat-square
[github-release-link]: https://github.com/open-compass/opencompass/releases
[github-release-shield]: https://img.shields.io/github/v/release/open-compass/opencompass?color=369eff&labelColor=black&logo=github&style=flat-square
[github-releasedate-link]: https://github.com/open-compass/opencompass/releases
[github-releasedate-shield]: https://img.shields.io/github/release-date/open-compass/opencompass?labelColor=black&style=flat-square
[github-stars-link]: https://github.com/open-compass/opencompass/stargazers
[github-stars-shield]: https://img.shields.io/github/stars/open-compass/opencompass?color=ffcb47&labelColor=black&style=flat-square
[github-trending-shield]: https://trendshift.io/api/badge/repositories/6630
[github-trending-url]: https://trendshift.io/repositories/6630
