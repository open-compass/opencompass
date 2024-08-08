# GPQA

```bash
python3 run.py --models hf_internlm2_7b --datasets gpqa_ppl_6bf57a --debug
python3 run.py --models hf_internlm2_chat_7b --datasets gpqa_gen_4baadb --debug
```

## Base Models

|          model           |   GPQA_diamond |
|:------------------------:|---------------:|
|    llama-7b-turbomind    |          24.24 |
|   llama-13b-turbomind    |          25.25 |
|   llama-30b-turbomind    |          22.73 |
|   llama-65b-turbomind    |          21.72 |
|   llama-2-7b-turbomind   |          25.25 |
|  llama-2-13b-turbomind   |          23.74 |
|  llama-2-70b-turbomind   |          28.28 |
|   llama-3-8b-turbomind   |          31.82 |
|  llama-3-70b-turbomind   |          40.91 |
| internlm2-1.8b-turbomind |          24.24 |
|  internlm2-7b-turbomind  |          28.28 |
| internlm2-20b-turbomind  |          31.31 |
|   qwen-1.8b-turbomind    |          28.79 |
|    qwen-7b-turbomind     |          24.75 |
|    qwen-14b-turbomind    |          27.78 |
|    qwen-72b-turbomind    |          31.31 |
|     qwen1.5-0.5b-hf      |          23.74 |
|     qwen1.5-1.8b-hf      |          28.79 |
|      qwen1.5-4b-hf       |          23.23 |
|      qwen1.5-7b-hf       |          20.71 |
|      qwen1.5-14b-hf      |          32.32 |
|      qwen1.5-32b-hf      |          30.81 |
|      qwen1.5-72b-hf      |          31.82 |
|   qwen1.5-moe-a2-7b-hf   |          28.79 |
|    mistral-7b-v0.1-hf    |          24.75 |
|    mistral-7b-v0.2-hf    |          23.74 |
|   mixtral-8x7b-v0.1-hf   |          28.79 |
|  mixtral-8x22b-v0.1-hf   |          36.36 |
|         yi-6b-hf         |          28.28 |
|        yi-34b-hf         |          35.86 |
|   deepseek-7b-base-hf    |          20.71 |
|   deepseek-67b-base-hf   |          25.25 |

## Chat Models

|             model             |   GPQA_diamond |
|:-----------------------------:|---------------:|
|     qwen1.5-0.5b-chat-hf      |          19.70 |
|     qwen1.5-1.8b-chat-hf      |          29.80 |
|      qwen1.5-4b-chat-hf       |          25.25 |
|      qwen1.5-7b-chat-hf       |          31.82 |
|      qwen1.5-14b-chat-hf      |          30.30 |
|      qwen1.5-32b-chat-hf      |          31.31 |
|      qwen1.5-72b-chat-hf      |          32.83 |
|     qwen1.5-110b-chat-hf      |          35.86 |
|    internlm2-chat-1.8b-hf     |          25.76 |
|  internlm2-chat-1.8b-sft-hf   |          26.26 |
|     internlm2-chat-7b-hf      |          28.28 |
|   internlm2-chat-7b-sft-hf    |          27.27 |
|     internlm2-chat-20b-hf     |          30.30 |
|   internlm2-chat-20b-sft-hf   |          29.29 |
|    llama-3-8b-instruct-hf     |          25.76 |
|    llama-3-70b-instruct-hf    |          37.88 |
| llama-3-8b-instruct-lmdeploy  |          25.76 |
| llama-3-70b-instruct-lmdeploy |          37.88 |
|  mistral-7b-instruct-v0.1-hf  |          30.30 |
|  mistral-7b-instruct-v0.2-hf  |          25.25 |
| mixtral-8x7b-instruct-v0.1-hf |          30.30 |
