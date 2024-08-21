# RACE

```bash
python3 run.py --models hf_internlm2_7b --datasets race_ppl_abed12 --debug
python3 run.py --models hf_internlm2_chat_7b --datasets race_gen_69ee4f --debug
```

## Base Models

|          model           |   race-high |   race-middle |
|:------------------------:|------------:|--------------:|
|    llama-7b-turbomind    |       31.30 |         29.53 |
|   llama-13b-turbomind    |       35.56 |         40.25 |
|   llama-30b-turbomind    |       57.35 |         55.78 |
|   llama-65b-turbomind    |       70.21 |         75.35 |
|   llama-2-7b-turbomind   |       39.74 |         46.73 |
|  llama-2-13b-turbomind   |       57.06 |         60.52 |
|  llama-2-70b-turbomind   |       79.02 |         82.17 |
|   llama-3-8b-turbomind   |       67.75 |         73.61 |
|  llama-3-70b-turbomind   |       85.79 |         90.25 |
| internlm2-1.8b-turbomind |       64.72 |         70.40 |
|  internlm2-7b-turbomind  |       72.56 |         74.16 |
| internlm2-20b-turbomind  |       72.90 |         74.03 |
|   qwen-1.8b-turbomind    |       63.09 |         69.29 |
|    qwen-7b-turbomind     |       80.30 |         85.38 |
|    qwen-14b-turbomind    |       88.11 |         92.06 |
|    qwen-72b-turbomind    |       90.62 |         93.59 |
|     qwen1.5-0.5b-hf      |       54.66 |         60.38 |
|     qwen1.5-1.8b-hf      |       67.27 |         73.33 |
|      qwen1.5-4b-hf       |       78.50 |         83.29 |
|      qwen1.5-7b-hf       |       82.73 |         86.70 |
|      qwen1.5-14b-hf      |       87.99 |         91.85 |
|      qwen1.5-32b-hf      |       90.57 |         93.25 |
|      qwen1.5-72b-hf      |       90.45 |         93.87 |
|   qwen1.5-moe-a2-7b-hf   |       79.56 |         83.57 |
|    mistral-7b-v0.1-hf    |       73.58 |         76.25 |
|    mistral-7b-v0.2-hf    |       73.67 |         77.09 |
|   mixtral-8x7b-v0.1-hf   |       80.13 |         84.61 |
|  mixtral-8x22b-v0.1-hf   |       86.56 |         89.62 |
|         yi-6b-hf         |       82.93 |         85.72 |
|        yi-34b-hf         |       90.94 |         92.76 |
|   deepseek-7b-base-hf    |       50.91 |         56.82 |
|   deepseek-67b-base-hf   |       83.53 |         88.23 |

## Chat Models

|             model             |   race-high |   race-middle |
|:-----------------------------:|------------:|--------------:|
|     qwen1.5-0.5b-chat-hf      |       49.03 |         52.79 |
|     qwen1.5-1.8b-chat-hf      |       66.24 |         72.91 |
|      qwen1.5-4b-chat-hf       |       73.53 |         80.29 |
|      qwen1.5-7b-chat-hf       |       83.28 |         88.09 |
|      qwen1.5-14b-chat-hf      |       87.51 |         91.36 |
|      qwen1.5-32b-chat-hf      |       91.22 |         93.52 |
|      qwen1.5-72b-chat-hf      |       91.11 |         93.38 |
|     qwen1.5-110b-chat-hf      |       92.31 |         93.66 |
|    internlm2-chat-1.8b-hf     |       73.87 |         81.13 |
|  internlm2-chat-1.8b-sft-hf   |       73.81 |         81.69 |
|     internlm2-chat-7b-hf      |       84.51 |         88.72 |
|   internlm2-chat-7b-sft-hf    |       84.88 |         89.90 |
|     internlm2-chat-20b-hf     |       88.02 |         91.43 |
|   internlm2-chat-20b-sft-hf   |       88.11 |         91.57 |
|    llama-3-8b-instruct-hf     |       81.22 |         86.63 |
|    llama-3-70b-instruct-hf    |       89.57 |         93.45 |
| llama-3-8b-instruct-lmdeploy  |       81.02 |         86.14 |
| llama-3-70b-instruct-lmdeploy |       89.34 |         93.25 |
|  mistral-7b-instruct-v0.1-hf  |       69.75 |         74.72 |
|  mistral-7b-instruct-v0.2-hf  |       73.30 |         77.58 |
| mixtral-8x7b-instruct-v0.1-hf |       81.88 |         87.26 |
