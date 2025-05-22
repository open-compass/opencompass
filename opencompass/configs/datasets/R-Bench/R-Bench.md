# R-Bench

## Introduction

The following introduction comes from the description on the [R-Bench official website](https://evalmodels.github.io/rbench/):

```
R-Bench is a graduate-level multi-disciplinary benchmark for evaluating the complex reasoning capabilities of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). R stands for Reasoning.
```

According to statistics on R-Bench, the benchmark spans 19 departments, including mathematics, physics, biology, computer science, and chemistry, covering over 100 subjects such as Inorganic Chemistry, Chemical Reaction Kinetics, and Electromagnetism. It features 1,094 questions designed for testing language models and 665 questions specifically tailored for evaluating multimodal reasoning capabilities, available in both English and Chinese.

These questions are meticulously curated to ensure rigorous difficulty calibration, subject balance, and cross-linguistic alignment, enabling the assessment to be an Olympiad-level multi-disciplinary benchmark.

## Official Links

### Paper

[R-Bench: Graduate-level Multi-disciplinary Benchmarks for LLM & MLLM Complex Reasoning Evaluation](https://arxiv.org/abs/2505.02018)

## Evaluation Results

### Language Model Results

```
Model                     Source                                              Date       Average     RBench-T    RBench-T (zh)
------------------------  --------------------------------------------------  ----------  ----------  ----------  ---------------
OpenAI o1 🥇              https://openai.com/o1/                              2024-12-17  69.6        69.0        70.1
Gemini2.0-Flash-Thinking 🥈 https://deepmind.google/technologies/gemini/flash-thinking/ 2025-01-21 68.0 68.4      67.5
Doubao1.5Pro 🥉           https://www.volcengine.com/product/doubao           2025-01-21  62.7        62.0        63.4
GPT-4o                    https://openai.com/index/hello-gpt-4o/              2024-11-20  52.6        53.6        51.6
Claude3.5-sonnet          https://www.anthropic.com/news/claude-3-5-sonnet    2024-06-20  57.4        57.5        57.3
Qwen2.5-72B               https://github.com/QwenLM/Qwen2.5                   2024-09-19  52.9        53.7        52.0
Qwen2.5-32B               https://github.com/QwenLM/Qwen2.5                   2024-09-19  50.4        50.8        49.9
Qwen2.5-7B                https://github.com/QwenLM/Qwen2.5                   2024-09-19  44.1        43.6        44.5
```

### Multimodal Model Results

```
Model                     Source                                              Date       Average     RBench-M    RBench-M (zh)
------------------------  --------------------------------------------------  ----------  ----------  ----------  ---------------
OpenAI o1 🥇              https://openai.com/o1/                              2024-12-17  53.1        53.2        53.0
Doubao1.5Pro 🥈           https://www.volcengine.com/product/doubao           2025-01-21  40.2        37.9        42.4
Claude-3-5-sonnet 🥉      https://www.anthropic.com/news/claude-3-5-sonnet    2025-04-10  39.0        39.7        38.3
GPT-4o                    https://openai.com/index/hello-gpt-4o/              2024-11-20  33.3        33.4        33.2
Qwen2.5-72B               https://github.com/QwenLM/Qwen2.5                   2024-09-19  25.4        25.1        25.7
Qwen2.5-7B                https://github.com/QwenLM/Qwen2.5                   2024-09-19  21.0        19.6        22.3
```

Note:
- RBench-T: Text-only questions for language models test
- RBench-M: Multimodal questions for multimodal models test
- The values in the table represent the Top-1 accuracy, in %
- zh indicates the Chinese version

## Reference

```
@inproceedings{
  guo2025rbench,
  title={RBench: Graduate-level Multi-disciplinary Benchmarks for
    LLM & MLLM Complex Reasoning Evaluation},
  author={Meng-Hao Guo, Jiajun Xu, Yi Zhang, Jiaxi Song, Haoyang Peng, Yi-Xuan Deng, 
    Xinzhi Dong, Kiyohiro Nakayama, Zhengyang Geng, Chen Wang, Bolin Ni, Guo-Wei Yang, 
    Yongming Rao, Houwen Peng, Han Hu, Gordon Wetzstein, Shi-min Hu},
  year={2025},
  eprint={2505.02018},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2505.02018}, 
}
