# Prompt Overview

The prompt is the input to the Language Model (LLM), used to guide the model to generate text or calculate perplexity (PPL). The selection of prompts can significantly impact the accuracy of the evaluated model. The process of converting the dataset into a series of prompts is defined by templates.

In OpenCompass, we split the template into two parts: the data-side template and the model-side template. When evaluating a model, the data will pass through both the data-side template and the model-side template, ultimately transforming into the input required by the model.

The data-side template is referred to as [prompt_template](./prompt_template.md), which represents the process of converting the fields in the dataset into prompts.

The model-side template is referred to as [meta_template](./meta_template.md), which represents how the model transforms these prompts into its expected input.

We also offer some prompting examples regarding [Chain of Thought](./chain_of_thought.md).
