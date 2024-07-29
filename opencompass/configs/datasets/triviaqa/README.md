# TriviaQA

```bash
python3 run.py --models hf_internlm2_7b --datasets triviaqa_wiki_1shot_gen_20a989 --debug
python3 run.py --models hf_internlm2_chat_7b --datasets triviaqa_wiki_1shot_gen_eaf81e --debug
```

## Base Models

|          model           |   triviaqa |
|:------------------------:|-----------:|
|    llama-7b-turbomind    |      40.39 |
|   llama-13b-turbomind    |      66.41 |
|   llama-30b-turbomind    |      75.90 |
|   llama-65b-turbomind    |      82.26 |
|   llama-2-7b-turbomind   |      43.21 |
|  llama-2-13b-turbomind   |      71.32 |
|  llama-2-70b-turbomind   |      67.45 |
|   llama-3-8b-turbomind   |      71.24 |
|  llama-3-70b-turbomind   |      88.16 |
| internlm2-1.8b-turbomind |      38.42 |
|  internlm2-7b-turbomind  |      69.15 |
| internlm2-20b-turbomind  |      74.03 |
|   qwen-1.8b-turbomind    |      22.76 |
|    qwen-7b-turbomind     |      53.61 |
|    qwen-14b-turbomind    |      49.72 |
|    qwen-72b-turbomind    |      79.13 |
|     qwen1.5-0.5b-hf      |      21.24 |
|     qwen1.5-1.8b-hf      |      34.32 |
|      qwen1.5-4b-hf       |      44.59 |
|      qwen1.5-7b-hf       |      56.60 |
|      qwen1.5-14b-hf      |      59.96 |
|      qwen1.5-32b-hf      |      56.20 |
|      qwen1.5-72b-hf      |      77.81 |
|   qwen1.5-moe-a2-7b-hf   |      65.49 |
|    mistral-7b-v0.1-hf    |      72.93 |
|    mistral-7b-v0.2-hf    |      70.91 |
|   mixtral-8x7b-v0.1-hf   |      85.05 |
|  mixtral-8x22b-v0.1-hf   |      89.47 |
|         yi-6b-hf         |      23.76 |
|        yi-34b-hf         |      14.73 |
|   deepseek-7b-base-hf    |      59.48 |
|   deepseek-67b-base-hf   |      72.15 |

## Chat Models

|             model             |   triviaqa |
|:-----------------------------:|-----------:|
|     qwen1.5-0.5b-chat-hf      |      19.84 |
|     qwen1.5-1.8b-chat-hf      |      35.81 |
|      qwen1.5-4b-chat-hf       |      48.93 |
|      qwen1.5-7b-chat-hf       |      53.65 |
|      qwen1.5-14b-chat-hf      |      62.58 |
|      qwen1.5-32b-chat-hf      |      74.72 |
|      qwen1.5-72b-chat-hf      |      83.25 |
|     qwen1.5-110b-chat-hf      |      86.20 |
|    internlm2-chat-1.8b-hf     |      46.69 |
|  internlm2-chat-1.8b-sft-hf   |      46.50 |
|     internlm2-chat-7b-hf      |      69.54 |
|   internlm2-chat-7b-sft-hf    |      70.75 |
|     internlm2-chat-20b-hf     |      75.53 |
|   internlm2-chat-20b-sft-hf   |      75.90 |
|    llama-3-8b-instruct-hf     |      78.99 |
|    llama-3-70b-instruct-hf    |      89.79 |
| llama-3-8b-instruct-lmdeploy  |      76.77 |
| llama-3-70b-instruct-lmdeploy |      89.62 |
|  mistral-7b-instruct-v0.1-hf  |      62.94 |
|  mistral-7b-instruct-v0.2-hf  |      67.72 |
| mixtral-8x7b-instruct-v0.1-hf |      79.57 |
