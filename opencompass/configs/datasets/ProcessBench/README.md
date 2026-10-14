# ProcessBench: Identifying Process Errors in Mathematical Reasoning

ProcessBench is a benchmark dataset for evaluating large language models' ability to identify errors in mathematical reasoning processes. Developed by the Qwen team and published at **ACL 2025**, this dataset aims to assess whether models can accurately locate where errors occur in multi-step mathematical reasoning.

## Overview

### Purpose

While large language models excel at mathematical problem-solving, their ability to identify and locate errors in reasoning processes is equally important. ProcessBench focuses on evaluating models' capabilities in:
- **Error Detection**: Determining whether errors exist in the reasoning process
- **Error Localization**: Accurately identifying the step where the earliest error occurs
- **Process Understanding**: Deep comprehension of logical chains in multi-step mathematical reasoning

### Dataset Construction

ProcessBench contains four subsets at different difficulty levels:
- **gsm8k**: Grade school math word problems
- **math**: High school competition mathematics
- **olympiadbench**: Mathematical Olympiad problems
- **omnimath**: Comprehensive mathematical problems

Each sample includes:
- **problem**: The original math problem
- **steps**: Step-by-step solution process
- **label**: Index of the error step (-1 indicates no error)
- **final_answer_correct**: Whether the final answer is correct

## Usage

### 1. Data Preview

You can preview ProcessBench data using the following code:

```python
import json
from datasets import load_dataset

dataset = load_dataset('Qwen/ProcessBench', split='gsm8k')
print(json.dumps(dataset[0], indent=2))
```

### 2. Evaluation with OpenCompass

To evaluate models on ProcessBench using OpenCompass:

```bash
python run.py examples/eval_ProcessBench.py
```

### 3. Configuration

The dataset configuration is located at `opencompass/configs/datasets/ProcessBench/processbench_gen.py`, supporting the following subsets:
- `processbench_gsm8k`
- `processbench_math`
- `processbench_olympiadbench`
- `processbench_omnimath`

### 4. Evaluation Metrics

ProcessBench uses the following metrics to evaluate model performance:
- **error_acc**: Accuracy on error samples (whether the model can accurately locate the error position)
- **correct_acc**: Accuracy on correct samples (whether the model can identify error-free reasoning)
- **f1**: Harmonic mean of error_acc and correct_acc, providing a comprehensive measure of performance

## Citation

```bibtex
@inproceedings{processbench,
  title={ProcessBench: Identifying Process Errors in Mathematical Reasoning}, 
  author={Chujie Zheng and Zhenru Zhang and Beichen Zhang and Runji Lin and Keming Lu and
          Bowen Yu and Dayiheng Liu and Jingren Zhou and Junyang Lin},
  booktitle={The 63rd Annual Meeting of the Association for Computational Linguistics},
  year={2025}
}
```

## Resources

- 📄 [Paper](https://arxiv.org/abs/2412.06559)
- 🤗 [HuggingFace Dataset](https://huggingface.co/datasets/Qwen/ProcessBench)
- 💻 [Official Code Repository](https://github.com/QwenLM/ProcessBench)

