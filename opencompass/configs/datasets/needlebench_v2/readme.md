# NeedleBench V2: An Enhanced Benchmark for Needle-In-A-Haystack Evaluations

English | [简体中文](readme_zh-CN.md)

## Overview

NeedleBench V2 is an improved benchmark that rigorously assesses the information retrieval and reasoning capabilities of large language models (LLMs) in long-context scenarios. Building upon the original NeedleBench, this version introduces significant enhancements to provide more accurate and unbiased evaluations of LLMs' abilities to locate and reason with critical information in extensive texts.

### Directory Structure

```
configs/datasets/needlebench_v2/
├── atc
├── needlebench_v2_4k
├── needlebench_v2_8k
├── needlebench_v2_32k
├── needlebench_v2_128k
├── needlebench_v2_200k
├── needlebench_v2_256k
├── needlebench_v2_1000k
├── readme.md
└── readme_zh-CN.md
```

Within each configuration directory (e.g., `needlebench_v2_4k`), there are configuration files tailored for testing within that specific length setting.

## Task Descriptions and Length Configurations

NeedleBench V2 offers tasks in various length configurations (4k, 8k, 32k, 128k, 200k, 256k, 1000k) to accommodate different scales of language model evaluation needs. Each length configuration provides specialized test scripts for the following tasks:

### Single-Needle Retrieval

The Single-Needle Retrieval task evaluates LLMs' ability to recall a single piece of crucial information from a haystack text of a specific length. This task assesses the model's precision in identifying and recalling specific information from extended texts.

### Multi-Needle Retrieval

The Multi-Needle Retrieval task challenges LLMs' ability to identify and extract multiple key information points from extensive texts. It simulates real-world scenarios where multiple data points, facts, or figures need to be retrieved from documents or reports, evaluating the model's efficiency in navigating and extracting relevant information from dense texts.

### Multi-Needle Reasoning

In NeedleBench V2, the Multi-Needle Reasoning task has been significantly improved. The original needles based on the R4C/MultiHop dataset have been replaced with fictional information similar to those in the Ancestral Trace Challenge. This change addresses potential biases from innate knowledge, as the original dataset may have been included in some models' training data. The task continues to evaluate LLMs' capacity for complex reasoning with retrieved information, requiring models to not only recall multiple pieces of information but also engage in logical reasoning.

### Ancestral Trace Challenge (ATC)

The Ancestral Trace Challenge has been refined in NeedleBench V2. The needle distribution pattern has changed from a dense form (1, 2, 3, 4, 5 needles) to a sparse form based on powers of 2 (2¹, 2², 2³, etc.). This task remains NeedleBench's most complex, requiring models to recall and analyze every detail in long texts for problems demanding an understanding of complex relationships, such as genealogical inquiries or detailed case analysis.

## Scoring Methodology

NeedleBench V2 introduces a more balanced scoring system. The overall score is now calculated as a simple average of the three main tasks (Single-Needle Retrieval, Multi-Needle Retrieval, and Multi-Needle Reasoning), with each task receiving equal weight. This change from the previous weighted average approach provides a more straightforward and equitable assessment of model capabilities across different retrieval and reasoning tasks.

## Prompt Enhancements

All prompts in NeedleBench V2 have been refined for greater clarity and effectiveness, with particular attention to the ATC experiment prompts. The configuration structure has also been streamlined for easier use and interpretation.

## Citation

If you use NeedleBench V2 in your research, please cite:

```bibtex
@misc{li2025needlebenchllmsretrievalreasoning,
      title={NeedleBench: Can LLMs Do Retrieval and Reasoning in Information-Dense Context?}, 
      author={Mo Li and Songyang Zhang and Taolin Zhang and Haodong Duan and Yunxin Liu and Kai Chen},
      year={2025},
      eprint={2407.11963},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.11963}, 
}
``` 