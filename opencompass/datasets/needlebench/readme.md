# Needlebench: A Benchmark for Advanced Language Model Evaluation

English | [简体中文](readme_zh-CN.md)

## Overview

Needlebench is a comprehensive benchmark suite designed to rigorously evaluate the information retrieval and reasoning capabilities of large language models (LLMs). Drawing inspiration from the NeedleInAHaystack experiment, Needlebench broadens the scope to include a variety of tasks, each aimed at testing different facets of LLMs' abilities to process, recall, and reason with information embedded in lengthy texts.

### Directory Structure

```
opencompass/datasets/needlebench/
├── atc.py          # Ancestral Trace Challenge
├── multi.py        # Multi-Needles Reasoning
├── origin.py       # Single-Needle Retrieval
├── parallel.py     # Multi-Needles Retrieval
└── readme.md
```

## Task Descriptions

### Single-Needle Retrieval (`origin.py`)

The Single-Needle Retrieval task is foundational to the Needlebench suite, focusing on the LLM's ability to recall a single piece of crucial information from a haystack text of a specific length. This task mirrors the original NeedleInAHaystack test, assessing the model's precision in identifying and recalling specific information within large text bodies.

### Multi-Needles Retrieval (`parallel.py`)

The Multi-Needles Retrieval task challenges the LLM's ability to identify and extract multiple key pieces of information from extensive texts. It simulates real-world scenarios where multiple data points, facts, or figures need to be retrieved from documents or reports, evaluating the model's efficiency in navigating and extracting relevant information from dense texts.

### Multi-Needles Reasoning (`multi.py`)

Building on the retrieval tasks, the Multi-Needles Reasoning task emphasizes the LLM's capacity for complex reasoning with the retrieved information. The model must not only recall multiple pieces of information but also engage in logical reasoning, synthesizing answers that reflect an understanding of the relationships between various information points.

### Ancestral Trace Challenge (ATC) (`atc.py`)

The Ancestral Trace Challenge is the most complex task in the Needlebench suite, requiring models to recall and analyze every detail in long texts for problem-solving that demands an understanding of complex relationships, such as genealogical inquiries or detailed case analysis. This task highlights the need for models to process and reason with information at a detailed level, mirroring the demands of sophisticated real-world analytical tasks.
