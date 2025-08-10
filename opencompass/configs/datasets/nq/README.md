# NQ

```bash
python3 run.py --models hf_internlm2_7b --datasets nq_open_1shot_gen_20a989 --debug
python3 run.py --models hf_internlm2_chat_7b --datasets nq_open_1shot_gen_01cf41 --debug
```

## Base Models

|          model           |    nq |
|:------------------------:|------:|
|    llama-7b-turbomind    | 15.10 |
|   llama-13b-turbomind    | 16.43 |
|   llama-30b-turbomind    | 22.11 |
|   llama-65b-turbomind    | 26.09 |
|   llama-2-7b-turbomind   | 14.35 |
|  llama-2-13b-turbomind   | 21.69 |
|  llama-2-70b-turbomind   | 23.27 |
|   llama-3-8b-turbomind   | 18.78 |
|  llama-3-70b-turbomind   | 31.88 |
| internlm2-1.8b-turbomind | 20.66 |
|  internlm2-7b-turbomind  | 41.05 |
| internlm2-20b-turbomind  | 43.55 |
|   qwen-1.8b-turbomind    |  5.68 |
|    qwen-7b-turbomind     | 17.87 |
|    qwen-14b-turbomind    | 13.77 |
|    qwen-72b-turbomind    | 18.20 |
|     qwen1.5-0.5b-hf      |  6.01 |
|     qwen1.5-1.8b-hf      | 10.28 |
|      qwen1.5-4b-hf       | 15.73 |
|      qwen1.5-7b-hf       | 18.61 |
|      qwen1.5-14b-hf      | 16.07 |
|      qwen1.5-32b-hf      | 21.75 |
|      qwen1.5-72b-hf      | 20.53 |
|   qwen1.5-moe-a2-7b-hf   | 16.62 |
|    mistral-7b-v0.1-hf    | 20.66 |
|    mistral-7b-v0.2-hf    | 20.78 |
|   mixtral-8x7b-v0.1-hf   | 24.85 |
|  mixtral-8x22b-v0.1-hf   | 34.43 |
|         yi-6b-hf         | 10.08 |
|        yi-34b-hf         | 13.96 |
|   deepseek-7b-base-hf    |  8.45 |
|   deepseek-67b-base-hf   | 17.59 |

## Chat Models

|             model             |    nq |
|:-----------------------------:|------:|
|     qwen1.5-0.5b-chat-hf      |  7.42 |
|     qwen1.5-1.8b-chat-hf      | 10.22 |
|      qwen1.5-4b-chat-hf       | 19.31 |
|      qwen1.5-7b-chat-hf       | 16.87 |
|      qwen1.5-14b-chat-hf      | 20.53 |
|      qwen1.5-32b-chat-hf      | 25.26 |
|      qwen1.5-72b-chat-hf      | 35.21 |
|     qwen1.5-110b-chat-hf      | 36.98 |
|    internlm2-chat-1.8b-hf     | 19.09 |
|  internlm2-chat-1.8b-sft-hf   | 18.14 |
|     internlm2-chat-7b-hf      | 28.73 |
|   internlm2-chat-7b-sft-hf    | 30.78 |
|     internlm2-chat-20b-hf     | 28.75 |
|   internlm2-chat-20b-sft-hf   | 34.10 |
|    llama-3-8b-instruct-hf     | 30.17 |
|    llama-3-70b-instruct-hf    | 40.25 |
| llama-3-8b-instruct-lmdeploy  | 28.28 |
| llama-3-70b-instruct-lmdeploy | 39.14 |
|  mistral-7b-instruct-v0.1-hf  | 22.47 |
|  mistral-7b-instruct-v0.2-hf  | 25.18 |
| mixtral-8x7b-instruct-v0.1-hf | 32.08 |
