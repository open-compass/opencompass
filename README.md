<div align="center">
  <img src="docs/en/_static/image/logo.svg" width="500px"/>
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

[🌐Website](https://opencompass.org.cn/) |
[📖CompassHub](https://hub.opencompass.org.cn/home) |
[📊CompassRank](https://rank.opencompass.org.cn/home) |
[📘Documentation](https://opencompass.readthedocs.io/en/latest/) |
[🛠️Installation](https://opencompass.readthedocs.io/en/latest/get_started/installation.html) |
[🤔Reporting Issues](https://github.com/open-compass/opencompass/issues/new/choose)

English | [简体中文](README_zh-CN.md)

[![][github-trending-shield]][github-trending-url]

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/KKwfEbFj7U" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=opencompass" target="_blank">WeChat</a>
</p>

> \[!IMPORTANT\]
>
> **Star Us**, You will receive all release notifications from GitHub without any delay ~ ⭐️

<details>
  <summary><kbd>Star History</kbd></summary>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=open-compass%2Fopencompass&theme=dark&type=Date">
    <img width="100%" src="https://api.star-history.com/svg?repos=open-compass%2Fopencompass&type=Date">
  </picture>
</details>

## 🧭	Welcome

to **OpenCompass**!

Just like a compass guides us on our journey, OpenCompass will guide you through the complex landscape of evaluating large language models. With its powerful algorithms and intuitive interface, OpenCompass makes it easy to assess the quality and effectiveness of your NLP models.

🚩🚩🚩 Explore opportunities at OpenCompass! We're currently **hiring full-time researchers/engineers and interns**. If you're passionate about LLM and OpenCompass, don't hesitate to reach out to us via [email](mailto:opencompass@pjlab.org.cn). We'd love to hear from you!

🔥🔥🔥 We are delighted to announce that **the OpenCompass has been recommended by the Meta AI**, click [Get Started](https://ai.meta.com/llama/get-started/#validation) of Llama for more information.

## ✨ Introduction

![image](https://github.com/open-compass/opencompass/assets/22607038/f45fe125-4aed-4f8c-8fe8-df4efb41a8ea)

OpenCompass is a one-stop platform for large model evaluation. It supports a wide range of models from OpenAI, Anthropic, Gemini, Qwen, GLM, DeepSeek, and more, and integrates over 100 datasets covering knowledge, reasoning, coding, science, language, long context, safety, and other capability dimensions. OpenCompass also integrates VLMEvalKit, enabling text-only and vision-language datasets to be evaluated within a unified workflow.

OpenCompass provides a complete workflow spanning dataset and model configuration, inference, evaluation, and result summarization. It supports local open-source models and API models, as well as multiple inference backends such as HuggingFace, LMDeploy, and vLLM. The platform supports zero-shot, few-shot, chain-of-thought, rule-based, and LLM-as-judge evaluation, while task partitioning, concurrent execution, and distributed execution accommodate evaluation workloads of different scales.

OpenCompass is designed to be open, reproducible, and extensible. Users can flexibly integrate new models, datasets, evaluators, inference backends, and task scheduling systems. The accompanying [CompassHub](https://hub.opencompass.org.cn/home) provides benchmark navigation, while [CompassRank](https://rank.opencompass.org.cn/home) presents model evaluation leaderboards.

## 🚀 What's New <a><img width="35" height="20" src="https://user-images.githubusercontent.com/12782558/212848161-5e783dd6-11e8-4fe0-bbba-39ffb77730be.png"></a>

- **\[2026.09.21\]** OpenCompass has updated the recommended API configurations for flagship models from leading providers and comprehensively restructured its Chinese and English documentation, further improving the model integration and evaluation experience. See the [model configurations](opencompass/configs/models/) and [OpenCompass documentation](docs/en/index.rst) for details! 🔥🔥🔥
- **\[2026.08.25\]** OpenCompass now integrates with [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), enabling native multimodal dataset loading, inference through OpenAI-compatible APIs, and evaluation with official VLMEvalKit metrics. Check out the [MMBench example](examples/eval_mmbench_vlmevalkit.py) and [MMMU-Pro example](examples/eval_mmmu_pro_vlmevalkit.py) for details! 🔥🔥🔥
- **\[2026.07.28\]** OpenCompass has expanded its API model ecosystem with support for the OpenAI Responses API and LiteLLM AI Gateway, while updating the Gemini and Anthropic integrations to their latest SDK interfaces. Check out the [OpenAI Responses API implementation](opencompass/models/openai_response.py), [LiteLLM AI Gateway implementation](opencompass/models/litellm_api.py), [Gemini SDK implementation](opencompass/models/gemini_sdk_api.py), and [Anthropic SDK implementation](opencompass/models/claude_sdk_api.py) for details!
- **\[2026.07.27\]** OpenCompass now supports multi-round inference in `GenInferencer` and adds support for the Multi-IF dataset to evaluate multi-turn instruction-following capabilities. Check out the [Multi-IF evaluation configuration](opencompass/configs/datasets/MultiIF/MultiIF_gen.py) for details! 🔥🔥🔥
- **\[2026.05.25\]** OpenCompass now provides repeat analysis tools for detecting repetitive content and looping model outputs in current evaluation tasks or existing evaluation results. Check out the [repeat analysis tool](tools/analyze_repeat.py) for details!
- **\[2026.03.20\]** OpenCompass now supports concurrent inference across tasks together with evaluation watching, enabling completed inference tasks to be monitored and subsequent evaluations to be triggered in a coordinated pipeline. Parallel inferencers, task monitoring, and heartbeat mechanisms further improve large-scale evaluation efficiency. Check out the [concurrent inference implementation](opencompass/tasks/openicl_infer_concurrent.py) and [evaluation watcher implementation](opencompass/tasks/openicl_eval_watch.py) for details!
- **\[2026.03.17\]** OpenCompass introduces `RawPromptTemplate`, allowing original benchmark prompts and structured conversations to be passed to models without unintended formatting transformations. It supports API models, ChatML datasets, and appending additional prompt content on the model side. Check out the [RawPromptTemplate guide](docs/en/prompt/raw_prompt_template.md) for details!
- **\[2026.02.05\]** OpenCompass now supports Intern-S1-Pro related general and scientific evaluation benchmarks. Please check [Example for Evaluating Intern-S1-Pro](examples/eval_intern_s1_pro.py) and [Model Card](https://huggingface.co/internlm/Intern-S1-Pro) for more details! 🔥🔥🔥
- **\[2025.12.08\]** OpenCompass now supports evaluation for SciReasoner. Please check [Example for Evaluating SciReasoner](examples/eval_scireasoner.py) and [Project GitHub Repo](https://github.com/InternScience/SciReason) for more details! 🔥🔥🔥
- **\[2025.07.26\]** OpenCompass now supports Intern-S1 related general and scientific evaluation benchmarks. Please refer to the [Intern-S1 model configuration](opencompass/configs/models/intern_s/intern_s1.py) for details! 🔥🔥🔥
- **\[2025.04.01\]** OpenCompass now supports `CascadeEvaluator`, allowing multiple evaluators to work in sequence and enabling custom evaluation pipelines for more complex scenarios. Check out the [documentation](docs/en/evaluation/cascade_evaluator.md) for details! 🔥🔥🔥
- **\[2025.03.11\]** OpenCompass now supports `SuperGPQA`, covering knowledge evaluation across 285 graduate-level disciplines. Give it a try! 🔥🔥🔥
- **\[2025.02.28\]** OpenCompass now supports the `DeepSeek-R1` model series. Check out the [DeepSeek-R1 model configuration](opencompass/configs/models/deepseek/deepseek_r1_streaming.py) for more details! 🔥🔥🔥
- **\[2025.02.15\]** We have added two practical evaluation tools: `GenericLLMEvaluator` for LLM-as-judge evaluation and `MATHVerifyEvaluator` for mathematical reasoning evaluation. Check out the [LLM Judge](docs/en/evaluation/llm_judge.md) and [Mathematical Evaluation](docs/en/evaluation/math_verify.md) documentation for more details! 🔥🔥🔥

## 🛠️ Installation

Below are the steps for quick installation and dataset preparation.

### 💻 Environment Setup

We highly recommend using `conda` to manage your Python environment. OpenCompass supports
Python 3.12 for regular and full installations. If your evaluation depends on code execution
datasets backed by `pyext`, use Python 3.10 instead: `pyext==0.7` is skipped on Python >=3.11
because it relies on `inspect.getargspec`, which was removed in Python 3.11. Without `pyext`,
APPS (`apps`, `apps_mini`), TACO, and LiveCodeBench Code Generation are unavailable.

- #### Create your virtual environment

  ```bash
  conda create --name opencompass python=3.12 -y
  conda activate opencompass
  ```

- #### Install OpenCompass via pip

  ```bash
  # Supports most datasets and models
  pip install -U opencompass

  # Full installation (supports more datasets)
  # pip install "opencompass[full]"

  # Model inference backends. Since these backends often have conflicting dependencies,
  # we recommend managing them in separate virtual environments.
  # pip install "opencompass[lmdeploy]"
  # pip install "opencompass[vllm]"

  # API evaluation (for example, OpenAI and Qwen)
  # pip install "opencompass[api]"

  # Multimodal evaluation
  # pip install "opencompass[vlm]"
  ```

- #### Install OpenCompass from source

  To use the latest OpenCompass features, you can also build it from source:

  ```bash
  git clone https://github.com/open-compass/opencompass opencompass
  cd opencompass
  pip install -e .
  # pip install -e ".[full]"
  # pip install -e ".[vllm]"
  ```

After installation, read the [Quick Start](https://opencompass.readthedocs.io/en/latest/get_started/quick_start.html) to learn how to run an evaluation task.

For more tutorials, see our [full documentation](https://opencompass.readthedocs.io/en/latest/index.html).

<p align="right"><a href="#top">🔝Back to top</a></p>

## 📖 Dataset Support

The OpenCompass documentation provides a statistical list of all datasets supported by the platform.

You can quickly find the dataset you need by sorting, filtering, and searching the list.

See the dataset statistics section of the [dataset documentation](docs/en/user_guides/datasets.md) for details.

<p align="right"><a href="#top">🔝Back to top</a></p>

## 📖 Model Support

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

- [DeepSeek](opencompass/configs/models/deepseek/)
- [Gemma](opencompass/configs/models/gemma/)
- [GLM](opencompass/configs/models/glm/)
- [Intern-S](opencompass/configs/models/intern_s/)
- [Kimi](opencompass/configs/models/moonshot/)
- [Llama](opencompass/configs/models/hf_llama/)
- [MiniMax](opencompass/configs/models/minimax/)
- [Mistral](opencompass/configs/models/mistral/)
- [Qwen](opencompass/configs/models/qwen3/)
- [More model configurations](opencompass/configs/models/)

</td>
<td>

- [OpenAI](opencompass/configs/models/openai/)
- [Anthropic Claude](opencompass/configs/models/claude/)
- [Google Gemini](opencompass/configs/models/gemini/)
- [ByteDance Doubao](opencompass/configs/models/bytedance/)
- [xAI Grok](opencompass/configs/models/xai/)
- [More model configurations](opencompass/configs/models/)

</td>

</tr>
  </tbody>
</table>

<p align="right"><a href="#top">🔝Back to top</a></p>

## 📊 Leaderboard

We will continue to provide detailed leaderboards for open-source and API models. See the [OpenCompass Leaderboard](https://rank.opencompass.org.cn/home). To participate in an evaluation, send the model repository URL or a standard API endpoint via [email](mailto:opencompass@pjlab.org.cn).

<p align="right"><a href="#top">🔝Back to top</a></p>

## 👷‍♂️ Contributing

We appreciate all contributions to improving OpenCompass. Please refer to the [contributing guideline](https://opencompass.readthedocs.io/en/latest/faq/contribution_guide.html) for the best practice.

<a href="https://github.com/open-compass/opencompass/graphs/contributors" target="_blank">
  <table>
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=open-compass/opencompass"><br><br>
      </th>
    </tr>
  </table>
</a>

## 🤝 Acknowledgements

Some code in this project is cited and modified from [OpenICL](https://github.com/Shark-NLP/OpenICL).

Some datasets and prompt implementations are modified from [chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub) and [instruct-eval](https://github.com/declare-lab/instruct-eval).

## 🖊️ Citation

```bibtex
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}
```

<p align="right"><a href="#top">🔝Back to top</a></p>

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
