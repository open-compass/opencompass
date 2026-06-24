# IFEval

```bash
python3 run.py --models hf_internlm2_chat_7b --datasets IFEval_gen_3321a3 --debug
```

## Chat Models

|             model             |   Prompt-level-strict-accuracy |   Inst-level-strict-accuracy |   Prompt-level-loose-accuracy |   Inst-level-loose-accuracy |
|:-----------------------------:|-------------------------------:|-----------------------------:|------------------------------:|----------------------------:|
|     qwen1.5-0.5b-chat-hf      |                          13.12 |                        23.26 |                         15.71 |                       26.38 |
|     qwen1.5-1.8b-chat-hf      |                          16.08 |                        26.26 |                         18.30 |                       29.02 |
|      qwen1.5-4b-chat-hf       |                          25.51 |                        35.97 |                         28.84 |                       39.81 |
|      qwen1.5-7b-chat-hf       |                          38.82 |                        50.00 |                         42.70 |                       53.48 |
|      qwen1.5-14b-chat-hf      |                          42.51 |                        54.20 |                         49.17 |                       59.95 |
|      qwen1.5-32b-chat-hf      |                          49.54 |                        60.43 |                         53.97 |                       64.39 |
|      qwen1.5-72b-chat-hf      |                          51.02 |                        61.99 |                         57.12 |                       67.27 |
|     qwen1.5-110b-chat-hf      |                          55.08 |                        65.59 |                         61.18 |                       70.86 |
|    internlm2-chat-1.8b-hf     |                          18.30 |                        28.78 |                         21.44 |                       32.01 |
|  internlm2-chat-1.8b-sft-hf   |                          18.67 |                        31.18 |                         19.78 |                       32.85 |
|     internlm2-chat-7b-hf      |                          34.75 |                        46.28 |                         40.48 |                       51.44 |
|   internlm2-chat-7b-sft-hf    |                          39.19 |                        50.12 |                         42.33 |                       52.76 |
|     internlm2-chat-20b-hf     |                          36.41 |                        48.68 |                         40.67 |                       53.24 |
|   internlm2-chat-20b-sft-hf   |                          44.55 |                        55.64 |                         46.77 |                       58.03 |
|    llama-3-8b-instruct-hf     |                          68.02 |                        76.74 |                         75.42 |                       82.85 |
|    llama-3-70b-instruct-hf    |                          78.00 |                        84.65 |                         84.29 |                       89.21 |
| llama-3-8b-instruct-lmdeploy  |                          69.13 |                        77.46 |                         77.26 |                       83.93 |
| llama-3-70b-instruct-lmdeploy |                          75.97 |                        82.97 |                         83.18 |                       88.37 |
|  mistral-7b-instruct-v0.1-hf  |                          40.30 |                        50.96 |                         41.96 |                       53.48 |
|  mistral-7b-instruct-v0.2-hf  |                          49.17 |                        60.43 |                         51.94 |                       64.03 |
| mixtral-8x7b-instruct-v0.1-hf |                          50.09 |                        60.67 |                         55.64 |                       65.83 |
