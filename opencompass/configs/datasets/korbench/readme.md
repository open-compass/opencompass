# KOR-Bench: Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks

KOR-Bench is a dataset designed to evaluate large language models (LLMs) on tasks that require reasoning independent of prior knowledge. Created to assess reasoning and planning abilities, KOR-Bench introduces rule-based tasks that minimize the influence of pretrained knowledge, enabling a focused evaluation of intrinsic model capabilities.

## Overview

### Purpose

Large language models, such as GPT-4 and Claude, excel in knowledge-based tasks but face challenges in applying reasoning skills to unfamiliar scenarios. KOR-Bench is built to evaluate such reasoning capabilities across five categories:
- **Operation**: Arithmetic and logical operations.
- **Logic**: Complex deductive and inductive reasoning.
- **Cipher**: Code-breaking and pattern discovery.
- **Puzzle**: Problem-solving with creative and logical reasoning.
- **Counterfactual**: Hypothetical reasoning in alternate scenarios.

### Dataset Construction

KOR-Bench tasks are designed with novel rules and configurations, ensuring no reliance on pretrained knowledge. Each task includes:
- **Rules**: Custom rule sets to guide reasoning.
- **Questions**: Carefully crafted problems that require the application of rules.
- **Evaluation Scenarios**: Zero-shot, three-shot, and subquestion-specific configurations.

The dataset is structured to assess multistep reasoning, pattern recognition, and adaptability to new rules.

### Dataset Access

KOR-Bench is publicly available with detailed usage instructions in the [GitHub Repository](https://github.com/KOR-Bench/KOR-Bench). Download the dataset and leverage predefined evaluation scripts or customize your own.

### Evaluation

1. Install dependencies and configure your environment.
2. Run evaluations using `opencompass examples/eval_korbench.py` to assess LLM performance.
3. Analyze model performance across various reasoning tasks.

### Example Command
```bash
opencompass examples/eval_korbench.py
```

## Baselines and Results
KOR-Bench includes baseline results for leading LLMs evaluated across various configurations, including zero-shot (gen) and few-shot modes. Below is a summary of the results.
| dataset | version | metric | mode | internlm2_5-7b-chat-turbomind | internlm2_5-1_8b-chat-turbomind | llama-3_1-8b-instruct-turbomind | glm-4-9b-chat-turbomind | gemma-2-9b-it-turbomind |
|---------|---------|--------|------|--------------------------------|---------------------------------|---------------------------------|--------------------------|--------------------------|
| korbench_mixed_Multi-Q | 21f998 | accuracy | gen | 0.60 | 0.20 | 9.60 | 8.70 | 7.80 |
| korbench_mixed_Multi-R | 21f998 | accuracy | gen | 1.70 | 0.10 | 8.80 | 12.10 | 9.80 |
| korbench_mixed_Multi-RQ | 21f998 | accuracy | gen | 1.50 | 0.10 | 6.40 | 8.60 | 6.00 |
| korbench_cipher | 21f998 | accuracy | gen | 8.80 | 0.80 | 14.00 | 6.80 | 6.40 |
| korbench_counterfactual | 21f998 | accuracy | gen | 83.60 | 17.20 | 88.80 | 90.40 | 87.60 |
| korbench_logic | 21f998 | accuracy | gen | 8.40 | 3.60 | 37.60 | 38.80 | 40.80 |
| korbench_operation | 21f998 | accuracy | gen | 56.00 | 25.20 | 68.40 | 63.60 | 67.60 |
| korbench_puzzle | 21f998 | accuracy | gen | 3.60 | 0.00 | 3.20 | 3.20 | 5.60 |
| korbench_cipher | 21f998 | accuracy | fewshot | 8.40 | 3.20 | 9.60 | 9.20 | 9.60 |
| korbench_counterfactual | 21f998 | accuracy | fewshot | 87.60 | 58.00 | 23.60 | 89.60 | 84.40 |
| korbench_logic | 21f998 | accuracy | fewshot | 45.20 | 19.60 | 24.40 | 38.40 | 54.00 |
| korbench_operation | 21f998 | accuracy | fewshot | 24.80 | 11.20 | 73.20 | 67.20 | 23.20 |
| korbench_puzzle | 21f998 | accuracy | fewshot | 4.80 | 2.40 | 1.60 | 3.60 | 6.80 |

### Citation

**BibTeX:**
```bibtex
@misc{ma2024korbenchbenchmarkinglanguagemodels,
title={KOR-Bench: Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks}, 
author={Kaijing Ma and Xinrun Du and Yunran Wang and Haoran Zhang and Zhoufutu Wen and Xingwei Qu and Jian Yang and Jiaheng Liu and Minghao Liu and Xiang Yue and Wenhao Huang and Ge Zhang},
year={2024},
eprint={2410.06526},
archivePrefix={arXiv},
primaryClass={cs.DB},
url={https://arxiv.org/abs/2410.06526}, 
}
```
