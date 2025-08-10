# HellaSwag

```bash
python3 run.py --models hf_internlm2_7b --datasets hellaswag_10shot_ppl_59c85e --debug
python3 run.py --models hf_internlm2_chat_7b --datasets hellaswag_10shot_gen_e42710 --debug
```

## Base Models

|          model           |   hellaswag |
|:------------------------:|------------:|
|    llama-7b-turbomind    |       26.99 |
|   llama-13b-turbomind    |       34.21 |
|   llama-30b-turbomind    |       35.65 |
|   llama-65b-turbomind    |       44.63 |
|   llama-2-7b-turbomind   |       29.29 |
|  llama-2-13b-turbomind   |       45.06 |
|  llama-2-70b-turbomind   |       55.91 |
|   llama-3-8b-turbomind   |       50.86 |
|  llama-3-70b-turbomind   |       80.60 |
| internlm2-1.8b-turbomind |       44.86 |
|  internlm2-7b-turbomind  |       89.52 |
| internlm2-20b-turbomind  |       91.41 |
|   qwen-1.8b-turbomind    |       38.04 |
|    qwen-7b-turbomind     |       64.62 |
|    qwen-14b-turbomind    |       85.88 |
|    qwen-72b-turbomind    |       90.40 |
|     qwen1.5-0.5b-hf      |       29.19 |
|     qwen1.5-1.8b-hf      |       42.32 |
|      qwen1.5-4b-hf       |       55.89 |
|      qwen1.5-7b-hf       |       68.51 |
|      qwen1.5-14b-hf      |       83.86 |
|      qwen1.5-32b-hf      |       87.28 |
|      qwen1.5-72b-hf      |       90.41 |
|   qwen1.5-moe-a2-7b-hf   |       72.42 |
|    mistral-7b-v0.1-hf    |       42.04 |
|    mistral-7b-v0.2-hf    |       46.24 |
|   mixtral-8x7b-v0.1-hf   |       66.22 |
|  mixtral-8x22b-v0.1-hf   |       79.66 |
|         yi-6b-hf         |       66.83 |
|        yi-34b-hf         |       83.83 |
|   deepseek-7b-base-hf    |       30.42 |
|   deepseek-67b-base-hf   |       70.75 |

## Chat Models

|             model             |   hellaswag |
|:-----------------------------:|------------:|
|     qwen1.5-0.5b-chat-hf      |       29.60 |
|     qwen1.5-1.8b-chat-hf      |       41.71 |
|      qwen1.5-4b-chat-hf       |       60.45 |
|      qwen1.5-7b-chat-hf       |       71.58 |
|      qwen1.5-14b-chat-hf      |       79.70 |
|      qwen1.5-32b-chat-hf      |       88.56 |
|      qwen1.5-72b-chat-hf      |       89.37 |
|     qwen1.5-110b-chat-hf      |       91.11 |
|    internlm2-chat-1.8b-hf     |       60.47 |
|  internlm2-chat-1.8b-sft-hf   |       61.58 |
|     internlm2-chat-7b-hf      |       84.80 |
|   internlm2-chat-7b-sft-hf    |       85.21 |
|     internlm2-chat-20b-hf     |       88.48 |
|   internlm2-chat-20b-sft-hf   |       88.95 |
|    llama-3-8b-instruct-hf     |       74.39 |
|    llama-3-70b-instruct-hf    |       89.07 |
| llama-3-8b-instruct-lmdeploy  |       73.31 |
| llama-3-70b-instruct-lmdeploy |       87.28 |
|  mistral-7b-instruct-v0.1-hf  |       53.00 |
|  mistral-7b-instruct-v0.2-hf  |       65.72 |
| mixtral-8x7b-instruct-v0.1-hf |       76.16 |
