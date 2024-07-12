# MBPP

```bash
python3 run.py --models hf_internlm2_7b --datasets sanitized_mbpp_gen_742f0c --debug
python3 run.py --models hf_internlm2_chat_7b --datasets sanitized_mbpp_mdblock_gen_a447ff --debug
```

## Base Models

|          model           |   pass@1 |   pass |   timeout |   failed |   wrong_answer |
|:------------------------:|---------:|-------:|----------:|---------:|---------------:|
|    llama-7b-turbomind    |    25.29 |     65 |         8 |       62 |            122 |
|   llama-13b-turbomind    |    29.96 |     77 |         4 |       74 |            102 |
|   llama-30b-turbomind    |    37.35 |     96 |        17 |       39 |            105 |
|   llama-65b-turbomind    |    45.53 |    117 |        10 |       35 |             95 |
|   llama-2-7b-turbomind   |    26.46 |     68 |        18 |       49 |            122 |
|  llama-2-13b-turbomind   |    36.58 |     94 |        17 |       45 |            101 |
|  llama-2-70b-turbomind   |    49.42 |    127 |        12 |       32 |             86 |
|   llama-3-8b-turbomind   |    54.86 |    141 |        11 |       22 |             83 |
|  llama-3-70b-turbomind   |    77.82 |    200 |         0 |       10 |             47 |
| internlm2-1.8b-turbomind |    30.74 |     79 |        10 |       61 |            107 |
|  internlm2-7b-turbomind  |    54.47 |    140 |        11 |       28 |             78 |
| internlm2-20b-turbomind  |    59.92 |    154 |         6 |       31 |             66 |
|   qwen-1.8b-turbomind    |     2.72 |      7 |        16 |      222 |             12 |
|    qwen-7b-turbomind     |    46.69 |    120 |        10 |       37 |             90 |
|    qwen-14b-turbomind    |    55.64 |    143 |         0 |       31 |             83 |
|    qwen-72b-turbomind    |    65.76 |    169 |         0 |       26 |             62 |
|     qwen1.5-0.5b-hf      |     5.06 |     13 |        13 |      190 |             41 |
|     qwen1.5-1.8b-hf      |    15.95 |     41 |        19 |      124 |             73 |
|      qwen1.5-4b-hf       |    45.91 |    118 |         8 |       27 |            104 |
|      qwen1.5-7b-hf       |    52.14 |    134 |        11 |       24 |             88 |
|      qwen1.5-14b-hf      |    52.14 |    134 |        16 |       33 |             74 |
|      qwen1.5-32b-hf      |    59.14 |    152 |         7 |       25 |             73 |
|      qwen1.5-72b-hf      |    61.09 |    157 |         1 |       21 |             78 |
|   qwen1.5-moe-a2-7b-hf   |    47.08 |    121 |         0 |       52 |             84 |
|    mistral-7b-v0.1-hf    |    47.47 |    122 |         9 |       33 |             93 |
|    mistral-7b-v0.2-hf    |    49.81 |    128 |         9 |       27 |             93 |
|   mixtral-8x7b-v0.1-hf   |    62.65 |    161 |        10 |       13 |             73 |
|  mixtral-8x22b-v0.1-hf   |    73.15 |    188 |         1 |       10 |             58 |
|         yi-6b-hf         |    30.35 |     78 |         8 |       40 |            131 |
|        yi-34b-hf         |    48.64 |    125 |         0 |       43 |             89 |
|   deepseek-7b-base-hf    |    43.97 |    113 |        11 |       34 |             99 |
|   deepseek-67b-base-hf   |    64.98 |    167 |         0 |       24 |             66 |

## Chat Models

|             model             |   pass@1 |   pass |   timeout |   failed |   wrong_answer |
|:-----------------------------:|---------:|-------:|----------:|---------:|---------------:|
|     qwen1.5-0.5b-chat-hf      |    11.28 |     29 |         1 |      129 |             98 |
|     qwen1.5-1.8b-chat-hf      |    22.57 |     58 |         2 |       70 |            127 |
|      qwen1.5-4b-chat-hf       |    43.58 |    112 |         1 |       33 |            111 |
|      qwen1.5-7b-chat-hf       |    50.58 |    130 |         0 |       35 |             92 |
|      qwen1.5-14b-chat-hf      |    56.03 |    144 |         0 |       24 |             89 |
|      qwen1.5-32b-chat-hf      |    65.37 |    168 |         2 |       13 |             74 |
|      qwen1.5-72b-chat-hf      |    66.93 |    172 |         0 |       17 |             68 |
|     qwen1.5-110b-chat-hf      |    68.48 |    176 |         0 |       16 |             65 |
|    internlm2-chat-1.8b-hf     |    39.69 |    102 |         0 |       48 |            107 |
|  internlm2-chat-1.8b-sft-hf   |    36.19 |     93 |         1 |       58 |            105 |
|     internlm2-chat-7b-hf      |    57.59 |    148 |         0 |       21 |             88 |
|   internlm2-chat-7b-sft-hf    |    55.64 |    143 |         2 |       22 |             90 |
|     internlm2-chat-20b-hf     |    68.87 |    177 |         0 |       16 |             64 |
|   internlm2-chat-20b-sft-hf   |    69.65 |    179 |         0 |       16 |             62 |
|    llama-3-8b-instruct-hf     |    68.87 |    177 |         0 |        8 |             72 |
|    llama-3-70b-instruct-hf    |    79.77 |    205 |         0 |        2 |             50 |
| llama-3-8b-instruct-lmdeploy  |    66.93 |    172 |         0 |        7 |             78 |
| llama-3-70b-instruct-lmdeploy |    77.82 |    200 |         1 |        2 |             54 |
|  mistral-7b-instruct-v0.1-hf  |    47.86 |    123 |         0 |       29 |            105 |
|  mistral-7b-instruct-v0.2-hf  |    45.91 |    118 |         0 |       31 |            108 |
| mixtral-8x7b-instruct-v0.1-hf |    61.48 |    158 |         1 |       13 |             85 |
