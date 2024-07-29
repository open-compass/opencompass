# MATH

```bash
python3 run.py --models hf_internlm2_7b --datasets math_4shot_base_gen_db136b --debug
python3 run.py --models hf_internlm2_chat_7b --datasets math_0shot_gen_393424 --debug
```

## Base Models

|          model           |   math |
|:------------------------:|-------:|
|    llama-7b-turbomind    |   2.94 |
|   llama-13b-turbomind    |   3.84 |
|   llama-30b-turbomind    |   6.54 |
|   llama-65b-turbomind    |  10.66 |
|   llama-2-7b-turbomind   |   3.58 |
|  llama-2-13b-turbomind   |   5.30 |
|  llama-2-70b-turbomind   |  13.26 |
|   llama-3-8b-turbomind   |  16.42 |
|  llama-3-70b-turbomind   |  39.64 |
| internlm2-1.8b-turbomind |   9.42 |
|  internlm2-7b-turbomind  |  25.16 |
| internlm2-20b-turbomind  |  32.24 |
|   qwen-1.8b-turbomind    |   6.30 |
|    qwen-7b-turbomind     |  15.56 |
|    qwen-14b-turbomind    |  30.38 |
|    qwen-72b-turbomind    |  44.18 |
|     qwen1.5-0.5b-hf      |   4.16 |
|     qwen1.5-1.8b-hf      |  11.32 |
|      qwen1.5-4b-hf       |  17.50 |
|      qwen1.5-7b-hf       |  17.34 |
|      qwen1.5-14b-hf      |  36.18 |
|      qwen1.5-32b-hf      |  45.74 |
|      qwen1.5-72b-hf      |  41.56 |
|   qwen1.5-moe-a2-7b-hf   |  27.96 |
|    mistral-7b-v0.1-hf    |  13.44 |
|    mistral-7b-v0.2-hf    |  12.74 |
|   mixtral-8x7b-v0.1-hf   |  29.46 |
|  mixtral-8x22b-v0.1-hf   |  41.82 |
|         yi-6b-hf         |   6.60 |
|        yi-34b-hf         |  18.80 |
|   deepseek-7b-base-hf    |   4.66 |
|   deepseek-67b-base-hf   |  18.76 |

## Chat Models

|             model             |   math |
|:-----------------------------:|-------:|
|     qwen1.5-0.5b-chat-hf      |   0.56 |
|     qwen1.5-1.8b-chat-hf      |   4.94 |
|      qwen1.5-4b-chat-hf       |   7.34 |
|      qwen1.5-7b-chat-hf       |  22.14 |
|      qwen1.5-14b-chat-hf      |  32.22 |
|      qwen1.5-32b-chat-hf      |  41.80 |
|      qwen1.5-72b-chat-hf      |  45.22 |
|     qwen1.5-110b-chat-hf      |  54.38 |
|    internlm2-chat-1.8b-hf     |  14.06 |
|  internlm2-chat-1.8b-sft-hf   |  13.10 |
|     internlm2-chat-7b-hf      |  28.08 |
|   internlm2-chat-7b-sft-hf    |  27.60 |
|     internlm2-chat-20b-hf     |  34.68 |
|   internlm2-chat-20b-sft-hf   |  32.54 |
|    llama-3-8b-instruct-hf     |  27.50 |
|    llama-3-70b-instruct-hf    |  47.52 |
| llama-3-8b-instruct-lmdeploy  |  27.42 |
| llama-3-70b-instruct-lmdeploy |  46.90 |
|  mistral-7b-instruct-v0.1-hf  |   8.48 |
|  mistral-7b-instruct-v0.2-hf  |  10.82 |
| mixtral-8x7b-instruct-v0.1-hf |  27.02 |
