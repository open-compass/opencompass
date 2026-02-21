# HumanEval

```bash
python3 run.py --models hf_internlm2_7b --datasets deprecated_humaneval_gen_d2537e --debug
python3 run.py --models hf_internlm2_chat_7b --datasets humaneval_gen_8e312c --debug
```

## Base Models

|          model           |   pass@1 |
|:------------------------:|---------:|
|    llama-7b-turbomind    |    12.80 |
|   llama-13b-turbomind    |    15.24 |
|   llama-30b-turbomind    |     9.15 |
|   llama-65b-turbomind    |     7.32 |
|   llama-2-7b-turbomind   |    14.02 |
|  llama-2-13b-turbomind   |    15.24 |
|  llama-2-70b-turbomind   |    15.24 |
|   llama-3-8b-turbomind   |    28.05 |
|  llama-3-70b-turbomind   |    28.05 |
| internlm2-1.8b-turbomind |    30.49 |
|  internlm2-7b-turbomind  |    48.17 |
| internlm2-20b-turbomind  |    51.83 |
|   qwen-1.8b-turbomind    |    16.46 |
|    qwen-7b-turbomind     |    23.78 |
|    qwen-14b-turbomind    |    23.78 |
|    qwen-72b-turbomind    |    66.46 |
|     qwen1.5-0.5b-hf      |     8.54 |
|     qwen1.5-1.8b-hf      |    23.17 |
|      qwen1.5-4b-hf       |    41.46 |
|      qwen1.5-7b-hf       |    53.05 |
|      qwen1.5-14b-hf      |    57.32 |
|      qwen1.5-32b-hf      |    70.12 |
|      qwen1.5-72b-hf      |    65.85 |
|   qwen1.5-moe-a2-7b-hf   |    45.73 |
|    mistral-7b-v0.1-hf    |    14.02 |
|    mistral-7b-v0.2-hf    |     9.15 |
|   mixtral-8x7b-v0.1-hf   |    24.39 |
|  mixtral-8x22b-v0.1-hf   |    16.46 |
|         yi-6b-hf         |    14.63 |
|        yi-34b-hf         |    17.07 |
|   deepseek-7b-base-hf    |    18.29 |
|   deepseek-67b-base-hf   |    23.17 |

## Chat Models

|             model             |   pass@1 |
|:-----------------------------:|---------:|
|     qwen1.5-0.5b-chat-hf      |     9.15 |
|     qwen1.5-1.8b-chat-hf      |    15.85 |
|      qwen1.5-4b-chat-hf       |    30.49 |
|      qwen1.5-7b-chat-hf       |    40.85 |
|      qwen1.5-14b-chat-hf      |    50.00 |
|      qwen1.5-32b-chat-hf      |    57.93 |
|      qwen1.5-72b-chat-hf      |    60.37 |
|     qwen1.5-110b-chat-hf      |    65.24 |
|    internlm2-chat-1.8b-hf     |    33.54 |
|  internlm2-chat-1.8b-sft-hf   |    34.15 |
|     internlm2-chat-7b-hf      |    56.71 |
|   internlm2-chat-7b-sft-hf    |    61.59 |
|     internlm2-chat-20b-hf     |    67.68 |
|   internlm2-chat-20b-sft-hf   |    67.68 |
|    llama-3-8b-instruct-hf     |    55.49 |
|    llama-3-70b-instruct-hf    |    70.73 |
| llama-3-8b-instruct-lmdeploy  |    57.93 |
| llama-3-70b-instruct-lmdeploy |    70.73 |
|  mistral-7b-instruct-v0.1-hf  |    32.32 |
|  mistral-7b-instruct-v0.2-hf  |    29.27 |
| mixtral-8x7b-instruct-v0.1-hf |    34.15 |
