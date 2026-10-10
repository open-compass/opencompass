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

🚩🚩🚩 欢迎加入 OpenCompass！我们目前**招聘全职研究人员/工程师和实习生**。如果您对 LLM 和 OpenCompass 充满热情，请随时通过[电子邮件](mailto:opencompass@pjlab.org.cn)与我们联系。我们非常期待与您交流！

🔥🔥🔥 祝贺 **OpenCompass 作为大模型标准测试工具被Meta AI官方推荐**, 点击 Llama 的 [入门文档](https://ai.meta.com/llama/get-started/#validation) 获取更多信息。

## ✨ 介绍

![image](https://github.com/open-compass/opencompass/assets/22607038/30bcb2e2-3969-4ac5-9f29-ad3f4abb4f3b)

OpenCompass 是面向大模型评测的一站式平台，支持来自 OpenAI、Anthropic、Gemini、Qwen、GLM、DeepSeek 等广泛的模型，集成超过 100 个数据集，覆盖知识、推理、代码、科学、语言、长上下文、安全等多个能力维度。OpenCompass 也已集成 VLMEvalKit，可在统一的评测流程中支持文本数据集与图文数据集。

OpenCompass 提供从数据集与模型配置、推理、评测到结果汇总的完整工作流，支持本地开源模型和 API 模型，以及 HuggingFace、LMDeploy、vLLM 等多种推理后端。平台支持零样本、少样本、思维链、基于规则及 LLM Judge 等多种评测方式，并可通过任务切分、并发与分布式执行满足不同规模的评测需求。

OpenCompass 坚持开放、可复现和易扩展的设计，用户可以灵活接入新的模型、数据集、评测器、推理后端及任务调度系统。配套的 [CompassHub](https://hub.opencompass.org.cn/home) 提供评测基准导航，[CompassRank](https://rank.opencompass.org.cn/home) 展示模型评测榜单。

## 🚀 最新进展 <a><img width="35" height="20" src="https://user-images.githubusercontent.com/12782558/212848161-5e783dd6-11e8-4fe0-bbba-39ffb77730be.png"></a>

- **\[2026.09.21\]** OpenCompass 更新了主流厂商旗舰模型的 API 推荐配置，并全面重构了中英文文档体系，进一步改善模型接入与评测体验。详情请参阅 [模型配置](opencompass/configs/models/)和 [OpenCompass 文档](docs/zh_cn/index.rst)！🔥🔥🔥
- **\[2026.08.25\]** OpenCompass 现已集成 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)，支持原生加载多模态数据集、通过 OpenAI 兼容接口完成推理，并使用 VLMEvalKit 官方评测逻辑计算指标。详情请参阅 [MMBench 评测示例](examples/eval_mmbench_vlmevalkit.py)和 [MMMU-Pro 评测示例](examples/eval_mmmu_pro_vlmevalkit.py)！🔥🔥🔥
- **\[2026.07.28\]** OpenCompass 进一步扩展了 API 模型生态，新增 OpenAI Responses API 和 LiteLLM AI Gateway 支持，并将 Gemini 与 Anthropic 集成更新至最新 SDK 接口。详情请参阅 [OpenAI Responses API 实现](opencompass/models/openai_response.py)、[LiteLLM AI Gateway 实现](opencompass/models/litellm_api.py)、[Gemini SDK 实现](opencompass/models/gemini_sdk_api.py)和 [Anthropic SDK 实现](opencompass/models/claude_sdk_api.py)！
- **\[2026.07.27\]** OpenCompass 的 `GenInferencer` 现已支持多轮推理，并新增 Multi-IF 数据集支持，用于评测模型的多轮指令遵循能力。详情请参阅 [Multi-IF 评测配置](opencompass/configs/datasets/MultiIF/MultiIF_gen.py)！🔥🔥🔥
- **\[2026.05.25\]** OpenCompass 新增重复输出分析工具，支持检测模型生成中的重复内容与循环输出，可用于分析当前评测任务或已有的评测结果。详情请参阅 [重复输出分析工具](tools/analyze_repeat.py)！
- **\[2026.03.20\]** OpenCompass 现已支持跨任务并发推理与评测监听，可协同监控已完成的推理任务并触发后续评测。并行 Inferencer、任务监控及心跳机制进一步提升了大规模评测效率。详情请参阅 [并发推理实现](opencompass/tasks/openicl_infer_concurrent.py) 和 [评测监听实现](opencompass/tasks/openicl_eval_watch.py)！
- **\[2026.03.17\]** OpenCompass 新增 `RawPromptTemplate`，可以在不引入非预期格式转换的情况下向模型传递 benchmark 的原始 Prompt 和结构化对话，并支持 API 模型、ChatML 数据集，以及模型侧的额外 Prompt 内容附加。详情请参阅 [RawPromptTemplate 使用指南](docs/zh_cn/prompt/raw_prompt_template.md)！
- **\[2026.02.05\]** OpenCompass 现已支持Intern-S1-Pro相关的通用及科学评测基准，请参阅[Intern-S1-Pro评测示例](examples/eval_intern_s1_pro.py)和[模型信息](https://huggingface.co/internlm/Intern-S1-Pro)了解详情！🔥🔥🔥
- **\[2025.12.08\]** OpenCompass 现已支持SciReasoner评测，请参阅[SciReasoner评测示例](examples/eval_scireasoner.py)和[原项目地址](https://github.com/InternScience/SciReason)了解详情！🔥🔥🔥
- **\[2025.07.26\]** OpenCompass 现已支持Intern-S1相关的通用及科学评测基准，请参阅[Intern-S1模型配置](opencompass/configs/models/intern_s/intern_s1.py)了解详情！🔥🔥🔥
- **\[2025.04.01\]** OpenCompass 现已支持 `CascadeEvaluator`，允许多个评估器按顺序工作，可以为更复杂的评估场景创建自定义评估流程，查看[文档](docs/zh_cn/evaluation/cascade_evaluator.md)了解具体用法！🔥🔥🔥
- **\[2025.03.11\]** 现已支持 `SuperGPQA`  覆盖285 个研究生学科的知识能力评测，欢迎尝试！🔥🔥🔥
- **\[2025.02.28\]** OpenCompass 现已支持 `DeepSeek-R1` 系列模型，请查看 [DeepSeek-R1 模型配置](opencompass/configs/models/deepseek/deepseek_r1_streaming.py) 了解更多详情！🔥🔥🔥
- **\[2025.02.15\]** 我们新增了两个实用的评测工具：用于 LLM 作为评判器的 `GenericLLMEvaluator` 和用于数学推理评估的 `MATHVerifyEvaluator`。查看 [LLM 评判器](docs/zh_cn/evaluation/llm_judge.md)和[数学能力评测](docs/zh_cn/evaluation/math_verify.md)文档了解更多详情！🔥🔥🔥

## 🛠️ 安装指南

下面提供了快速安装和数据集准备的步骤。

### 💻 环境搭建

我们强烈建议使用 `conda` 来管理您的 Python 环境。OpenCompass 的常规安装和完整安装已经支持
Python 3.12。如果您的评测依赖基于 `pyext` 的代码执行类数据集，请改用 Python 3.10：
`pyext==0.7` 会在 Python >=3.11 时被跳过，因为它依赖的 `inspect.getargspec` 已在
Python 3.11 中移除。缺少 `pyext` 时，APPS（`apps`、`apps_mini`）、TACO 和
LiveCodeBench Code Generation 将不可用。

- #### 创建虚拟环境

  ```bash
  conda create --name opencompass python=3.12 -y
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

  # 多模态评测
  # pip install "opencompass[vlm]"
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

完成安装后，接下来请阅读[快速开始](https://opencompass.readthedocs.io/zh_CN/latest/get_started/quick_start.html)了解如何运行一个评测任务。

更多教程请查看我们的[完整文档](https://opencompass.readthedocs.io/zh_CN/latest/index.html)。

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 📖 数据集支持

我们已经在OpenCompass官网的文档中支持了所有可在本平台上使用的数据集的统计列表。

您可以通过排序、筛选和搜索等功能从列表中快速找到您需要的数据集。

详情请参阅 [数据集相关文档](docs/zh_cn/user_guides/datasets.md) 的数据集统计章节。

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

- [DeepSeek](opencompass/configs/models/deepseek/)
- [Gemma](opencompass/configs/models/gemma/)
- [GLM](opencompass/configs/models/glm/)
- [Intern-S](opencompass/configs/models/intern_s/)
- [Kimi](opencompass/configs/models/moonshot/)
- [Llama](opencompass/configs/models/hf_llama/)
- [MiniMax](opencompass/configs/models/minimax/)
- [Mistral](opencompass/configs/models/mistral/)
- [Qwen](opencompass/configs/models/qwen3/)
- [更多模型配置](opencompass/configs/models/)

</td>
<td>

- [OpenAI](opencompass/configs/models/openai/)
- [Anthropic Claude](opencompass/configs/models/claude/)
- [Google Gemini](opencompass/configs/models/gemini/)
- [ByteDance Doubao](opencompass/configs/models/bytedance/)
- [xAI Grok](opencompass/configs/models/xai/)
- [更多模型配置](opencompass/configs/models/)

</td>

</tr>
  </tbody>
</table>

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 📊 性能榜单

我们将陆续提供开源模型和 API 模型的具体性能榜单，请见 [OpenCompass Leaderboard](https://rank.opencompass.org.cn/home) 。如需加入评测，请提供模型仓库地址或标准的 API 接口至[电子邮件](mailto:opencompass@pjlab.org.cn)。

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 👷‍♂️ 贡献

我们感谢所有的贡献者为改进和提升 OpenCompass 所作出的努力。请参考[贡献指南](https://opencompass.readthedocs.io/zh_CN/latest/faq/contribution_guide.html)来了解参与项目贡献的相关指引。

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
