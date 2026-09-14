Welcome to the OpenCompass Documentation
========================================

OpenCompass is a one-stop evaluation platform for large language models and
multimodal models. The documentation is organized so that you first complete
an evaluation, then understand and extend each component.

If this is your first time using OpenCompass, read `Installation and Environment
Setup`_, `Five-Minute Quick Start`_, and `Running a Complete Evaluation from a
Configuration`_ in order. If you already have a model or dataset, you can go
directly to the corresponding topic.

.. _Installation and Environment Setup:
.. _Five-Minute Quick Start:
.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/installation.md
   get_started/quick_start.md

.. _Running a Complete Evaluation from a Configuration:
.. toctree::
   :maxdepth: 1
   :caption: Basic Tutorials

   user_guides/framework_overview.md
   user_guides/config_based_evaluation.md
   user_guides/config.md
   user_guides/models.md
   user_guides/datasets.md
   user_guides/data_and_cache.md
   user_guides/results_and_summarizer.md

.. toctree::
   :maxdepth: 1
   :caption: Prompts and Input Construction

   prompt/raw_prompt_template.md
   prompt/meta_template.md
   prompt/chain_of_thought.md
   prompt/debugging.md

.. toctree::
   :maxdepth: 1
   :caption: Execution, Parallelism, and Task Management

   execution/tasks_and_runners.md
   execution/concurrent_evaluation.md
   execution/reuse_and_resume.md
   execution/cli_reference.md

.. toctree::
   :maxdepth: 1
   :caption: Evaluation Methods

   evaluation/metrics_and_postprocessing.md
   evaluation/llm_judge.md
   evaluation/math_verify.md
   evaluation/cascade_evaluator.md
   evaluation/code_eval.md
   evaluation/subjective_evaluation.md
   evaluation/repeated_evaluation.md
   evaluation/vlmevalkit.md

.. toctree::
   :maxdepth: 1
   :caption: Extending OpenCompass

   extension/extension_overview.md
   extension/new_dataset.md
   extension/custom_dataset.md
   extension/new_model.md
   extension/new_evaluator_and_summarizer.md
   extension/persistence.md

.. toctree::
   :maxdepth: 1
   :caption: Tools and Recovery

   tools/index.md
   tools/config_discovery.md
   tools/prediction_analysis.md
   tools/repeat_and_length.md
   tools/api_and_message_test.md
   tools/monitoring.md

.. toctree::
   :maxdepth: 1
   :caption: Reproduction, Contribution, and Releases

   notes/contribution_guide.md
   notes/academic.md

.. toctree::
   :maxdepth: 1
   :caption: Other Documentation

   faq/index.md
   faq/local_model_one_stop.md
   faq/long_context.md

Index and Search
================

* :ref:`genindex`
* :ref:`search`
