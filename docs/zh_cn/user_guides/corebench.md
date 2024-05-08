# 主要数据集性能

我们选择部分用于评估大型语言模型（LLMs）的知名基准，并提供了主要的LLMs在这些数据集上的详细性能结果。

| Model                | Version | Metric                       | Mode | GPT-4-1106 | GPT-4-0409 | Claude-3-Opus | Llama-3-70b-Instruct(lmdeploy) | Mixtral-8x22B-Instruct-v0.1 |
| -------------------- | ------- | ---------------------------- | ---- | ---------- | ---------- | ------------- | ------------------------------ | --------------------------- |
| MMLU                 | -       | naive_average                | gen  | 83.6       | 84.2       | 84.6          | 80.5                           | 77.2                        |
| CMMLU                | -       | naive_average                | gen  | 71.9       | 72.4       | 74.2          | 70.1                           | 59.7                        |
| CEval-Test           | -       | naive_average                | gen  | 69.7       | 70.5       | 71.7          | 66.9                           | 58.7                        |
| GaokaoBench          | -       | weighted_average             | gen  | 74.8       | 76.0       | 74.2          | 67.8                           | 60.0                        |
| Triviaqa_wiki(1shot) | 01cf41  | score                        | gen  | 73.1       | 82.9       | 82.4          | 89.8                           | 89.7                        |
| NQ_open(1shot)       | eaf81e  | score                        | gen  | 27.9       | 30.4       | 39.4          | 40.1                           | 46.8                        |
| Race-High            | 9a54b6  | accuracy                     | gen  | 89.3       | 89.6       | 90.8          | 89.4                           | 84.8                        |
| WinoGrande           | 6447e6  | accuracy                     | gen  | 80.7       | 83.3       | 84.1          | 69.7                           | 76.6                        |
| HellaSwag            | e42710  | accuracy                     | gen  | 92.7       | 93.5       | 94.6          | 87.7                           | 86.1                        |
| BBH                  | -       | naive_average                | gen  | 82.7       | 78.5       | 78.5          | 80.5                           | 79.1                        |
| GSM-8K               | 1d7fe4  | accuracy                     | gen  | 80.5       | 79.7       | 87.7          | 90.2                           | 88.3                        |
| Math                 | 393424  | accuracy                     | gen  | 61.9       | 71.2       | 60.2          | 47.1                           | 50                          |
| TheoremQA            | ef26ca  | accuracy                     | gen  | 28.4       | 23.3       | 29.6          | 25.4                           | 13                          |
| HumanEval            | 8e312c  | humaneval_pass@1             | gen  | 74.4       | 82.3       | 76.2          | 72.6                           | 72.0                        |
| MBPP(sanitized)      | 1e1056  | score                        | gen  | 78.6       | 77.0       | 76.7          | 71.6                           | 68.9                        |
| GPQA_diamond         | 4baadb  | accuracy                     | gen  | 40.4       | 48.5       | 46.5          | 38.9                           | 36.4                        |
| IFEval               | 3321a3  | Prompt-level-strict-accuracy | gen  | 71.9       | 79.9       | 80.0          | 77.1                           | 65.8                        |
