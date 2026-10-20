
# MuSR: Multistep Soft Reasoning Dataset

MuSR (Multistep Soft Reasoning) is a dataset designed to evaluate language models (LLMs) on complex reasoning tasks embedded in natural language narratives. Created to challenge state-of-the-art models like GPT-4 and others, MuSR emphasizes nuanced reasoning across different domains, including social and physical reasoning, commonsense reasoning, and planning, with tasks framed within realistic scenarios such as murder mysteries, object placements, and team allocations.

## Overview

### Purpose

Current large language models can perform complex tasks through prompting techniques like chain-of-thought reasoning. However, robust multistep reasoning remains challenging. MuSR addresses these limitations by evaluating LLM performance on tasks involving multistep reasoning in three domains:
- **Murder Mysteries**: Requires social and physical deductive reasoning.
- **Object Placements**: Tests observational and theory-of-mind reasoning.
- **Team Allocations**: Focuses on social reasoning and constraint satisfaction.

### Dataset Construction

MuSR instances are generated using a neurosymbolic synthetic-to-natural narrative generation algorithm. This approach allows for the creation of complex reasoning instances that combine structured reasoning trees with natural language narratives, challenging both direct and nuanced inference capabilities in LLMs.

MuSR's dataset consists of:
- **Murder Mysteries**: Scenarios with suspects, motives, and opportunities requiring deductive inference.
- **Object Placements**: Scenarios where individuals' observations inform reasoning about object locations.
- **Team Allocations**: Scenarios that simulate social relationships and teamwork for optimal task assignments.


### Dataset Access
MuSR dataset is publicly available, with instructions provided on the [GitHub Project](https://github.com/Zayne-Sprague/MuSR). You can download the dataset and use pre-defined prompts or create your own configurations.

### Evaluation

1. Install dependencies and configure the environment.
2. Run evaluations using `opencompass examples/eval_musr.py` to assess LLM performance.
3. Analyze results against human performance benchmarks.

### Example Command
```bash
opencompass examples/eval_musr.py
```

## Baselines and Results

MuSR includes baseline results for multiple LLMs evaluated with chain-of-thought and advanced reasoning strategies. These benchmarks assess model accuracy on reasoning tasks across the three domains.

| Domain           | Baseline Accuracy (GPT-4) | Human Performance |
|------------------|---------------------------|--------------------|
| Murder Mystery   | 80.4%                     | 94.1%             |
| Object Placement | 60.9%                     | 95.0%             |
| Team Allocation  | 68.4%                     | 100%              |


| dataset | version | metric | mode | internlm2_5-7b-chat-turbomind | qwen2.5-7b-instruct-turbomind | qwen2.5-14b-instruct-turbomind | yi-1.5-9b-chat-turbomind | qwen2.5-32b-instruct-turbomind | glm-4-9b-chat-turbomind | llama-3_1-8b-instruct-turbomind | ministral-8B-instruct-2410-turbomind | gemma-2-9b-it-turbomind | gemma-2-27b-it-turbomind |
|----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | -----|
| musr_murder_mysteries | a5ce30 | accuracy | gen | 59.20 | 63.20 | 76.00 | 68.80 | 78.80 | 71.20 | 73.60 | 73.60 | 74.80 | 77.20 |
| musr_object_placements | a5ce30 | accuracy | gen | 54.69 | 56.25 | 57.42 | 52.73 | 66.02 | 49.22 | 57.42 | 60.94 | 60.94 | 62.11 |
| musr_team_allocation | a5ce30 | accuracy | gen | 39.20 | 32.40 | 55.60 | 40.00 | 67.60 | 50.40 | 46.00 | 36.40 | 40.80 | 41.20 |
| musr_average | - | naive_average | gen | 51.03 | 50.62 | 63.01 | 53.84 | 70.81 | 56.94 | 59.01 | 56.98 | 58.85 | 60.17 |


## Citation

If you use MuSR in your research, please cite:
```bibtex
@misc{sprague2024musrtestinglimitschainofthought,
      title={MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning}, 
      author={Zayne Sprague and Xi Ye and Kaj Bostrom and Swarat Chaudhuri and Greg Durrett},
      year={2024},
      eprint={2310.16049},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.16049}, 
}
```

## Details

For further details, please refer to the MuSR paper [here](https://arxiv.org/abs/2310.16049).
