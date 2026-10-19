# LiveCodeBench

## Dataset

LiveCodeBench provides holistic and contamination-free evaluation of coding capabilities of LLMs. Particularly, LiveCodeBench continuously collects new problems over time from contests across three competition platforms -- LeetCode, AtCoder, and CodeForces. Next, LiveCodeBench also focuses on a broader range of code-related capabilities, such as self-repair, code execution, and test output prediction, beyond just code generation. Currently, LiveCodeBench hosts four hundred high-quality coding problems that were published between May 2023 and March 2024.

- Origin Project: https://livecodebench.github.io/leaderboard.html

## Setting

| Model Type | Code Generation | Test Output Prediction | Code Execution |
|------------|--------|--------|--------|
| Base Model    |  ❌      | ❌          |  ❌         |
| Chat Model    | ✅        |    ✅      |  ✅       |



## Baseline Performance


| Model Type | Code Generation(pass@1) | Test Output Prediction(pass@1) | Code Execution(pass@1) |
|------------|--------|--------|--------|
|  Qwen2.5-7B-Instruct(HF)    |  39.25      | 48.64          | 41.96         |
|  Meta-Llama-3.1-8B-Instruct(HF)  | 20.25       |   24.66     |  17.12      |


## Citation

```bibtex
@article{jain2024livecodebench,
  author    = {Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, Ion Stoica},
  title     = {LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code},
  year      = {2024},
  journal   = {arXiv preprint},
}
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}
```
