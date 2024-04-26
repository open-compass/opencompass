# Overview

## Evaluation Targets

The primary evaluation targets of this algorithm library are large language models. We introduce specific model types for evaluation using the large language model as an example.

- base Model: Typically obtained through training on massive textual data in a self-supervised manner (e.g., OpenAI's GPT-3, Meta's LLaMA). These models usually have powerful text continuation capabilities.

- Chat Model: Often built upon the base model and refined through directive fine-tuning or human preference alignment (e.g., OpenAI's ChatGPT, Shanghai AI Lab's Scholar Pu Tongue). These models can understand human instructions and have strong conversational skills.

## Tool Architecture

![framework-en](https://github.com/open-compass/opencompass/assets/17680578/b4d4bf4b-a673-4efe-b522-9337d4f7391a)

- Model Layer: This encompasses the primary model categories involved in large model evaluations. OpenCompass focuses on base models and chat models for in-depth evaluations.
- Capability Layer: OpenCompass evaluates models based on general capabilities and special features. In terms of general capabilities, models are evaluated on language, knowledge, understanding, reasoning, safety, and other dimensions. In terms of special capabilities, evaluations are based on long texts, code, tools, and knowledge enhancement.
- Method Layer: OpenCompass uses both objective and subjective evaluation methods. Objective evaluations can quickly assess a model's capability in tasks with definite answers (like multiple choice, fill in the blanks, closed-ended questions), while subjective evaluations measure user satisfaction with the model's replies. OpenCompass uses both model-assisted subjective evaluations and human feedback-driven subjective evaluations.
- Tool Layer: OpenCompass offers extensive functionalities for automated, efficient evaluations of large language models. This includes distributed evaluation techniques, prompt engineering, integration with evaluation databases, leaderboard publishing, report generation, and many more features.

## Capability Dimensions

### Design Philosophy

To accurately, comprehensively, and systematically assess the capabilities of large language models, OpenCompass takes a general AI perspective, integrating cutting-edge academic advancements and industrial best practices to propose an evaluation system tailored for real-world applications. OpenCompass's capability dimensions cover both general capabilities and special features.

### General Capabilities

General capabilities encompass examination, knowledge, language, understanding, reasoning, and safety, forming a comprehensive evaluation system across these six dimensions.

#### Examination Capability

This dimension aims to provide evaluation support from the perspective of human development, borrowing the classification logic from pedagogy. The core idea revolves around mandatory education, higher education, and vocational training, creating a comprehensive academic capability evaluation approach.

#### Knowledge Capability

Knowledge capability gauges the model's grasp on various knowledge types, including but not limited to general world knowledge and domain-specific expertise. This capability hopes that the model can answer a wide range of knowledge-based questions accurately and comprehensively.

#### Reasoning Capability

Reasoning is a crucial dimension for general AI. This evaluates the model's reasoning skills, including but not limited to mathematical computation, logical reasoning, causal inference, code generation and modification, and more.

#### Understanding Capability

This dimension evaluates the model's comprehension of text, including:

- Rhetorical techniques understanding and analysis: Grasping various rhetorical techniques used in text and analyzing and interpreting them.
- Text content summarization: Summarizing and extracting information from given content.
- Content creation: Open-ended or semi-open-ended content creation based on given themes or requirements.

#### Language Capability

This dimension evaluates the model's prior language knowledge, which includes but is not limited to:

- Word recognition and generation: Understanding language at the word level and tasks like word recognition, classification, definition, and generation.
- Grammar understanding and correction: Grasping grammar within the text and identifying and correcting grammatical errors.
- Cross-language translation: Translating given source language into target languages, assessing multilingual capabilities of current large models.

#### Safety Capability

In conjunction with the technical features of large language models, OpenCompass assesses the legality, compliance, and safety of model outputs, aiding the development of safe and responsible large models. This capability includes but is not limited to:

- Fairness
- Legality
- Harmlessness
- Ethical considerations
- Privacy protection

## Evaluation Methods

OpenCompass adopts a combination of objective and subjective evaluations. For capability dimensions and scenarios with definite answers, a comprehensive assessment of model capabilities is conducted using a well-constructed test set. For open-ended or semi-open-ended questions and model safety issues, a combination of objective and subjective evaluation methods is used.

### Objective Evaluation

For objective questions with standard answers, we can compare the discrepancy between the model's output and the standard answer using quantitative indicators. Given the high freedom in outputs of large language models, during evaluation, it's essential to standardize and design its inputs and outputs to minimize the influence of noisy outputs, ensuring a more comprehensive and objective assessment.

To better elicit the model's abilities in the evaluation domain and guide the model to output answers following specific templates, OpenCompass employs prompt engineering and in-context learning for objective evaluations.

In practice, we usually adopt the following two methods to evaluate model outputs:

- **Discriminative Evaluation**: This approach combines questions with candidate answers, calculates the model's perplexity on all combinations, and selects the answer with the lowest perplexity as the model's final output.

- **Generative Evaluation**: Used for generative tasks like language translation, code generation, logical analysis, etc. The question is used as the model's original input, leaving the answer area blank for the model to fill in. Post-processing of the output is often required to ensure it meets dataset requirements.

### Subjective Evaluation (Upcoming)

Language expression is lively and varied, and many scenarios and capabilities can't be judged solely by objective indicators. For evaluations like model safety and language capabilities, subjective evaluations based on human feelings better reflect the model's actual capabilities and align more with real-world applications.

OpenCompass's subjective evaluation approach relies on test subject's personal judgments to assess chat-capable large language models. In practice, we pre-construct a set of subjective test questions based on model capabilities and present different replies from various models to the same question to subjects, collecting their subjective scores. Given the high cost of subjective testing, this approach also uses high-performing large language models to simulate human subjective scoring. Actual evaluations will combine real human expert subjective evaluations with model-based subjective scores.

In conducting subjective evaluations, OpenCompass uses both **Single Model Reply Satisfaction Statistics** and **Multiple Model Satisfaction** Comparison methods.
