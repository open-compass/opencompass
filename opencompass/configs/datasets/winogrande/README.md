# WinoGrande

```bash
python3 run.py --models hf_internlm2_7b --datasets winogrande_5shot_ll_252f01 --debug
python3 run.py --models hf_internlm2_chat_7b --datasets winogrande_5shot_gen_b36770 --debug
```

## Base Models

|          model           |   winogrande |
|:------------------------:|-------------:|
|    llama-7b-turbomind    |        71.19 |
|   llama-13b-turbomind    |        76.16 |
|   llama-30b-turbomind    |        80.66 |
|   llama-65b-turbomind    |        82.16 |
|   llama-2-7b-turbomind   |        74.03 |
|  llama-2-13b-turbomind   |        76.48 |
|  llama-2-70b-turbomind   |        83.98 |
|   llama-3-8b-turbomind   |        77.82 |
|  llama-3-70b-turbomind   |        83.43 |
| internlm2-1.8b-turbomind |        66.77 |
|  internlm2-7b-turbomind  |        83.50 |
| internlm2-20b-turbomind  |        84.69 |
|   qwen-1.8b-turbomind    |        61.25 |
|    qwen-7b-turbomind     |        72.06 |
|    qwen-14b-turbomind    |        72.45 |
|    qwen-72b-turbomind    |        82.56 |
|     qwen1.5-0.5b-hf      |        57.38 |
|     qwen1.5-1.8b-hf      |        60.46 |
|      qwen1.5-4b-hf       |        65.90 |
|      qwen1.5-7b-hf       |        70.01 |
|      qwen1.5-14b-hf      |        72.93 |
|      qwen1.5-32b-hf      |        78.69 |
|      qwen1.5-72b-hf      |        80.74 |
|   qwen1.5-moe-a2-7b-hf   |        71.43 |
|    mistral-7b-v0.1-hf    |        78.30 |
|    mistral-7b-v0.2-hf    |        77.51 |
|   mixtral-8x7b-v0.1-hf   |        81.53 |
|  mixtral-8x22b-v0.1-hf   |        86.50 |
|         yi-6b-hf         |        74.35 |
|        yi-34b-hf         |        79.01 |
|   deepseek-7b-base-hf    |        74.11 |
|   deepseek-67b-base-hf   |        79.32 |

## Chat Models

|             model             |   winogrande |
|:-----------------------------:|-------------:|
|     qwen1.5-0.5b-chat-hf      |        50.51 |
|     qwen1.5-1.8b-chat-hf      |        51.07 |
|      qwen1.5-4b-chat-hf       |        57.54 |
|      qwen1.5-7b-chat-hf       |        65.27 |
|      qwen1.5-14b-chat-hf      |        70.09 |
|      qwen1.5-32b-chat-hf      |        77.90 |
|      qwen1.5-72b-chat-hf      |        80.82 |
|     qwen1.5-110b-chat-hf      |        82.32 |
|    internlm2-chat-1.8b-hf     |        57.62 |
|  internlm2-chat-1.8b-sft-hf   |        57.93 |
|     internlm2-chat-7b-hf      |        73.56 |
|   internlm2-chat-7b-sft-hf    |        73.80 |
|     internlm2-chat-20b-hf     |        81.06 |
|   internlm2-chat-20b-sft-hf   |        81.37 |
|    llama-3-8b-instruct-hf     |        66.22 |
|    llama-3-70b-instruct-hf    |        81.29 |
| llama-3-8b-instruct-lmdeploy  |        66.93 |
| llama-3-70b-instruct-lmdeploy |        81.22 |
|  mistral-7b-instruct-v0.1-hf  |        58.56 |
|  mistral-7b-instruct-v0.2-hf  |        59.43 |
| mixtral-8x7b-instruct-v0.1-hf |        65.75 |
