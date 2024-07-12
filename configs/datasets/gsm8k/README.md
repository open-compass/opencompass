# GSM8K

```bash
python3 run.py --models hf_internlm2_7b --datasets gsm8k_gen_17d0dc --debug
python3 run.py --models hf_internlm2_chat_7b --datasets gsm8k_gen_1d7fe4 --debug
```

## Base Models

|          model           |   gsm8k |
|:------------------------:|--------:|
|    llama-7b-turbomind    |   10.31 |
|   llama-13b-turbomind    |   20.55 |
|   llama-30b-turbomind    |   42.08 |
|   llama-65b-turbomind    |   54.81 |
|   llama-2-7b-turbomind   |   16.76 |
|  llama-2-13b-turbomind   |   29.87 |
|  llama-2-70b-turbomind   |   63.53 |
|   llama-3-8b-turbomind   |   54.28 |
|  llama-3-70b-turbomind   |   69.98 |
| internlm2-1.8b-turbomind |   30.40 |
|  internlm2-7b-turbomind  |   69.98 |
| internlm2-20b-turbomind  |   76.80 |
|   qwen-1.8b-turbomind    |   23.73 |
|    qwen-7b-turbomind     |   54.36 |
|    qwen-14b-turbomind    |   61.64 |
|    qwen-72b-turbomind    |   79.68 |
|     qwen1.5-0.5b-hf      |   13.27 |
|     qwen1.5-1.8b-hf      |   34.87 |
|      qwen1.5-4b-hf       |   47.61 |
|      qwen1.5-7b-hf       |   54.36 |
|      qwen1.5-14b-hf      |   63.53 |
|      qwen1.5-32b-hf      |   72.71 |
|      qwen1.5-72b-hf      |   79.53 |
|   qwen1.5-moe-a2-7b-hf   |   61.26 |
|    mistral-7b-v0.1-hf    |   47.61 |
|    mistral-7b-v0.2-hf    |   45.19 |
|   mixtral-8x7b-v0.1-hf   |   66.26 |
|  mixtral-8x22b-v0.1-hf   |   82.87 |
|         yi-6b-hf         |   39.58 |
|        yi-34b-hf         |   67.70 |
|   deepseek-7b-base-hf    |   20.17 |
|   deepseek-67b-base-hf   |   68.16 |

## Chat Models

|             model             |   gsm8k |
|:-----------------------------:|--------:|
|     qwen1.5-0.5b-chat-hf      |    8.79 |
|     qwen1.5-1.8b-chat-hf      |   27.60 |
|      qwen1.5-4b-chat-hf       |   47.61 |
|      qwen1.5-7b-chat-hf       |   56.25 |
|      qwen1.5-14b-chat-hf      |   64.90 |
|      qwen1.5-32b-chat-hf      |   79.91 |
|      qwen1.5-72b-chat-hf      |   77.03 |
|     qwen1.5-110b-chat-hf      |   79.53 |
|    internlm2-chat-1.8b-hf     |   39.73 |
|  internlm2-chat-1.8b-sft-hf   |   36.85 |
|     internlm2-chat-7b-hf      |   69.90 |
|   internlm2-chat-7b-sft-hf    |   69.83 |
|     internlm2-chat-20b-hf     |   75.21 |
|   internlm2-chat-20b-sft-hf   |   76.95 |
|    llama-3-8b-instruct-hf     |   79.53 |
|    llama-3-70b-instruct-hf    |   89.76 |
| llama-3-8b-instruct-lmdeploy  |   78.77 |
| llama-3-70b-instruct-lmdeploy |   89.31 |
|  mistral-7b-instruct-v0.1-hf  |   42.23 |
|  mistral-7b-instruct-v0.2-hf  |   45.56 |
| mixtral-8x7b-instruct-v0.1-hf |   65.13 |
