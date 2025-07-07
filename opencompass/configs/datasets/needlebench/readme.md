# Needlebench: A Benchmark for Needle-In-A-Haystack Evaluations

English | [简体中文](readme_zh-CN.md)

## Overview

Needlebench is an exhaustive benchmark designed to rigorously assess the information retrieval and reasoning capabilities of large language models (LLMs). Drawing inspiration from the NeedleInAHaystack experiment, Needlebench broadens the scope to include a variety of tasks, each aimed at testing different facets of LLMs' abilities in long-context scenarios.

### Directory Structure

```
configs/datasets/needlebench/
├── atc
├── needlebench_4k
├── needlebench_8k
├── needlebench_32k
├── needlebench_128k
├── needlebench_200k
├── needlebench_1000k
├── needlebench.py
├── readme.md
└── readme_zh-CN.md
```

Within each configuration directory (e.g., `needlebench_4k`), there are scripts tailored for testing within that specific length setting:

```
needlebench_4k/
├── needlebench_multi_reasoning.py
├── needlebench_multi_retrieval.py
├── needlebench.py
└── needlebench_single.py
```

## Task Descriptions and Length Configurations

Needlebench offers tasks in various length configurations, such as 4k, 8k, etc., to accommodate different scales of language model evaluation needs. Each length configuration provides specialized test scripts for the following tasks:

### Single-Needle Retrieval (`needlebench_single.py`)

The Single-Needle Retrieval task evaluates the LLMs' ability to recall a single piece of crucial information from a haystack text of a specific length. This task mirrors the original NeedleInAHaystack test's objective, assessing the model's precision in identifying and recalling specific information from extended texts.

### Multi-Needle Retrieval (`needlebench_multi_retrieval.py`)

The Multi-Needle Retrieval task challenges the LLMs' ability to identify and extract multiple key information points from extensive texts. It simulates real-world scenarios where multiple data points, facts, or figures need to be retrieved from documents or reports, evaluating the model's efficiency in navigating and extracting relevant information from dense texts.

### Multi-Needle Reasoning (`needlebench_multi_reasoning.py`)

Building on the retrieval tasks, the Multi-Needle Reasoning task emphasizes the LLMs' capacity for complex reasoning with the retrieved information. The model must not only recall multiple pieces of information but also engage in logical reasoning, synthesizing answers that reflect an understanding of the intricate relationships between various information points.

### Ancestral Trace Challenge (ATC)

The Ancestral Trace Challenge is Needlebench's most complex task, requiring models to recall and analyze every detail in long texts for problem-solving that demands an understanding of complex relationships, such as genealogical inquiries or detailed case analysis. This task highlights the need for models to process and reason with information at a granular level, mirroring the demands of sophisticated real-world analytical tasks.
