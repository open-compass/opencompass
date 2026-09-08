欢迎来到 OpenCompass 中文文档
================================

OpenCompass 是面向大语言模型与多模态模型的一站式评测平台。文档按照
“先完成一次评测，再理解和扩展各组件”的顺序组织。

如果你第一次使用 OpenCompass，请依次阅读 安装与环境准备_、五分钟快速开始_
和 使用配置文件完成一次完整评测_。如果你已经有模型或数据集，可以直接进入
对应专题。

.. _安装与环境准备:
.. _五分钟快速开始:
.. toctree::
   :maxdepth: 1
   :caption: 开始使用

   get_started/installation.md
   get_started/quick_start.md

.. _使用配置文件完成一次完整评测:
.. toctree::
   :maxdepth: 1
   :caption: 基础教程

   user_guides/framework_overview.md
   user_guides/config_based_evaluation.md
   user_guides/config.md
   user_guides/models.md
   user_guides/accelerator_intro.md
   user_guides/datasets.md
   user_guides/data_and_cache.md
   user_guides/results_and_summarizer.md

.. toctree::
   :maxdepth: 1
   :caption: 提示词与输入构造

   prompt/overview.md
   prompt/raw_prompt_template.md
   prompt/prompt_template.md
   prompt/meta_template.md
   prompt/chain_of_thought.md
   prompt/debugging.md

.. toctree::
   :maxdepth: 1
   :caption: 运行、并行与任务管理

   execution/tasks_and_runners.md
   execution/concurrent_evaluation.md
   execution/reuse_and_resume.md
   execution/cli_reference.md

.. toctree::
   :maxdepth: 1
   :caption: 评测方法

   evaluation/overview.md
   evaluation/metrics_and_postprocessing.md
   evaluation/llm_judge.md
   evaluation/math_verify.md
   evaluation/cascade_evaluator.md
   evaluation/code_eval.md
   evaluation/subjective_evaluation.md
   evaluation/repeated_evaluation.md
   evaluation/long_context.md
   evaluation/vlmevalkit.md

.. toctree::
   :maxdepth: 1
   :caption: 扩展 OpenCompass

   extension/extension_overview.md
   extension/new_dataset.md
   extension/custom_dataset.md
   extension/new_model.md
   extension/new_evaluator_and_summarizer.md
   extension/persistence.md

.. toctree::
   :maxdepth: 1
   :caption: 实用工具与故障恢复

   tools/index.md
   tools/config_discovery.md
   tools/prediction_analysis.md
   tools/repeat_and_length.md
   tools/api_and_message_test.md
   tools/monitoring.md

.. toctree::
   :maxdepth: 1
   :caption: 复现、贡献与版本说明

   notes/contribution_guide.md
   notes/academic.md

.. toctree::
   :maxdepth: 1
   :caption: 常见问题与故障排查

   faq/index.md

索引与搜索
============

* :ref:`genindex`
* :ref:`search`
