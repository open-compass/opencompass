# Extension Mechanisms Overview

OpenCompass builds its major components through registries and configuration. Before extending it, first identify the layer to which the requirement belongs:

| Requirement | Extension point |
| --- | --- |
| New data format or download logic | Dataset |
| New prompt, retrieval, or generation workflow | PromptTemplate / Retriever / Inferencer |
| New answer extraction or metric | Postprocessor / Evaluator |
| New model protocol or runtime | Model |
| New task partitioning or execution environment | Partitioner / Runner / Task |
| New leaderboard aggregation convention | Summarizer |

Prefer reusing existing components and adding configuration. Add a Python class only when the existing interfaces cannot express the required behavior. Extension code should be registered in the corresponding Registry and accompanied by a minimal configuration, unit tests, and relevant documentation.

- New data: see [Adding a Dataset](new_dataset.md) and [Quickly Evaluating Your Own Data](custom_dataset.md).
- New model: see [Adding a Model Backend](new_model.md).
- New evaluation component: see [Adding an Evaluator and Summarizer](new_evaluator_and_summarizer.md).
