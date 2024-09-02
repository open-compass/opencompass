# Qwen Model Details

## Qwen

Large language models (LLMs) have revolutionized the field of artificial intelligence, enabling natural language processing tasks that were previously thought to be exclusive to humans. In this work, we introduce Qwen, the first installment of our large language model series. Qwen is a comprehensive language model series that encompasses distinct models with varying parameter counts. It includes Qwen, the base pretrained language models, and Qwen-Chat, the chat models finetuned with human alignment techniques. The base language models consistently demonstrate superior performance across a multitude of downstream tasks, and the chat models, particularly those trained using Reinforcement Learning from Human Feedback (RLHF), are highly competitive. The chat models possess advanced tool-use and planning capabilities for creating agent applications, showcasing impressive performance even when compared to bigger models on complex tasks like utilizing a code interpreter. Furthermore, we have developed coding-specialized models, Code-Qwen and Code-Qwen-Chat, as well as mathematics-focused models, Math-Qwen-Chat, which are built upon base language models. These models demonstrate significantly improved performance in comparison with open-source models, and slightly fall behind the proprietary models.

## Qwen1.5

Qwen1.5 is the beta version of Qwen2, a transformer-based decoder-only language model pretrained on a large amount of data. In comparison with the previous released Qwen, the improvements include:

- 8 model sizes, including 0.5B, 1.8B, 4B, 7B, 14B, 32B and 72B dense models, and an MoE model of 14B with 2.7B activated;
- Significant performance improvement in human preference for chat models;
- Multilingual support of both base and chat models;
- Stable support of 32K context length for models of all sizes
- No need of trust_remote_code.

# Evaluation Command

## Base Models

```bash
python3 run.py --models hf_qwen1_5_7b --datasets mmlu_ppl_ac766d --debug
python3 run.py --models hf_qwen1_5_7b --datasets cmmlu_ppl_041cbf --debug
python3 run.py --models hf_qwen1_5_7b --datasets ceval_internal_ppl_93e5ce --debug
python3 run.py --models hf_qwen1_5_7b --datasets GaokaoBench_no_subjective_gen_d21e37 --debug
python3 run.py --models hf_qwen1_5_7b --datasets triviaqa_wiki_1shot_gen_20a989 --debug
python3 run.py --models hf_qwen1_5_7b --datasets nq_open_1shot_gen_20a989 --debug
python3 run.py --models hf_qwen1_5_7b --datasets race_ppl_abed12 --debug
python3 run.py --models hf_qwen1_5_7b --datasets winogrande_5shot_ll_252f01 --debug
python3 run.py --models hf_qwen1_5_7b --datasets hellaswag_10shot_ppl_59c85e --debug
python3 run.py --models hf_qwen1_5_7b --datasets bbh_gen_98fba6 --debug
python3 run.py --models hf_qwen1_5_7b --datasets gsm8k_gen_17d0dc --debug
python3 run.py --models hf_qwen1_5_7b --datasets math_4shot_base_gen_db136b --debug
python3 run.py --models hf_qwen1_5_7b --datasets TheoremQA_5shot_gen_6f0af8 --debug
python3 run.py --models hf_qwen1_5_7b --datasets deprecated_humaneval_gen_d2537e --debug
python3 run.py --models hf_qwen1_5_7b --datasets sanitized_mbpp_gen_742f0c --debug
python3 run.py --models hf_qwen1_5_7b --datasets lcbench_gen_5ff288 --debug
python3 run.py --models hf_qwen1_5_7b --datasets gpqa_ppl_6bf57a --debug
```

## Chat Models

```bash
python3 run.py --models hf_qwen1_5_7b_chat --datasets mmlu_gen_4d595a --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets cmmlu_gen_c13365 --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets ceval_internal_gen_2daf24 --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets GaokaoBench_no_subjective_gen_4c31db --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets triviaqa_wiki_1shot_gen_eaf81e --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets nq_open_1shot_gen_01cf41 --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets race_gen_69ee4f --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets winogrande_5shot_gen_b36770 --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets hellaswag_10shot_gen_e42710 --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets bbh_gen_5b92b0 --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets gsm8k_gen_1d7fe4 --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets math_0shot_gen_393424 --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets TheoremQA_5shot_gen_6f0af8 --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets humaneval_gen_8e312c --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets sanitized_mbpp_mdblock_gen_a447ff --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets lcbench_gen_5ff288 --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets gpqa_gen_4baadb --debug
python3 run.py --models hf_qwen1_5_7b_chat --datasets IFEval_gen_3321a3 --debug
```

# Benchmarks

We provide reference results for the classifical models, you can reproduce these results by following the aforementioned commands.

## Base Models

|   dataset    |   qwen-1.8b-turbomind |   qwen-7b-turbomind |   qwen-14b-turbomind |   qwen-72b-turbomind |
|:------------:|----------------------:|--------------------:|---------------------:|---------------------:|
|     mmlu     |                 46.61 |               59.75 |                67.85 |                77.36 |
|    cmmlu     |                 51.98 |               62.10 |                70.05 |                83.32 |
|  ceval-test  |                 54.24 |               62.06 |                70.33 |                83.25 |
| GaokaoBench  |                 22.11 |               35.32 |                54.07 |                77.56 |
|   triviaqa   |                 22.76 |               53.61 |                49.72 |                79.13 |
|      nq      |                  5.68 |               17.87 |                13.77 |                18.20 |
|  race-high   |                 63.09 |               80.30 |                88.11 |                90.62 |
|  winogrande  |                 61.25 |               72.06 |                72.45 |                82.56 |
|  hellaswag   |                 38.04 |               64.62 |                85.88 |                90.40 |
|     bbh      |                 22.53 |               45.89 |                56.75 |                63.35 |
|    gsm8k     |                 23.73 |               54.36 |                61.64 |                79.68 |
|     math     |                  6.30 |               15.56 |                30.38 |                44.18 |
|  TheoremQA   |                  9.38 |               15.00 |                21.62 |                27.12 |
|  humaneval   |                 16.46 |               23.78 |                23.78 |                66.46 |
|     mbpp     |                  2.72 |               46.69 |                55.64 |                65.76 |
|   lcbench    |                  1.82 |                4.95 |                 8.86 |                16.86 |
| GPQA_diamond |                 28.79 |               24.75 |                27.78 |                31.31 |

|   dataset    |   qwen1.5-0.5b-hf |   qwen1.5-1.8b-hf |   qwen1.5-4b-hf |   qwen1.5-7b-hf |   qwen1.5-14b-hf |   qwen1.5-32b-hf |   qwen1.5-72b-hf |
|:------------:|------------------:|------------------:|----------------:|----------------:|-----------------:|-----------------:|-----------------:|
|     mmlu     |             39.98 |             47.14 |           57.03 |           62.15 |            69.10 |            73.88 |            77.02 |
|    cmmlu     |             46.05 |             57.45 |           66.38 |           71.86 |            76.95 |            81.58 |            83.00 |
|  ceval-test  |             48.36 |             58.67 |           66.55 |           72.49 |            76.93 |            82.50 |            83.03 |
| GaokaoBench  |             30.67 |             35.66 |           54.31 |           65.99 |            66.60 |            79.01 |            80.26 |
|   triviaqa   |             21.24 |             34.32 |           44.59 |           56.60 |            59.96 |            56.20 |            77.81 |
|      nq      |              6.01 |             10.28 |           15.73 |           18.61 |            16.07 |            21.75 |            20.53 |
|  race-high   |             54.66 |             67.27 |           78.50 |           82.73 |            87.99 |            90.57 |            90.45 |
|  winogrande  |             57.38 |             60.46 |           65.90 |           70.01 |            72.93 |            78.69 |            80.74 |
|  hellaswag   |             29.19 |             42.32 |           55.89 |           68.51 |            83.86 |            87.28 |            90.41 |
|     bbh      |             20.54 |             27.01 |           34.81 |           39.87 |            50.38 |            67.47 |            58.81 |
|    gsm8k     |             13.27 |             34.87 |           47.61 |           54.36 |            63.53 |            72.71 |            79.53 |
|     math     |              4.16 |             11.32 |           17.50 |           17.34 |            36.18 |            45.74 |            41.56 |
|  TheoremQA   |              5.88 |             12.00 |           13.75 |            4.25 |            12.62 |            26.62 |            26.62 |
|  humaneval   |              8.54 |             23.17 |           41.46 |           53.05 |            57.32 |            70.12 |            65.85 |
|     mbpp     |              5.06 |             15.95 |           45.91 |           52.14 |            52.14 |            59.14 |            61.09 |
|   lcbench    |              0.87 |              2.00 |            5.65 |            6.69 |            12.69 |            14.34 |            15.29 |
| GPQA_diamond |             23.74 |             28.79 |           23.23 |           20.71 |            32.32 |            30.81 |            31.82 |

## Chat Models

|   dataset    |   qwen1.5-0.5b-chat-hf |   qwen1.5-1.8b-chat-hf |   qwen1.5-4b-chat-hf |   qwen1.5-7b-chat-hf |   qwen1.5-14b-chat-hf |   qwen1.5-32b-chat-hf |   qwen1.5-72b-chat-hf |   qwen1.5-110b-chat-hf |
|:------------:|-----------------------:|-----------------------:|---------------------:|---------------------:|----------------------:|----------------------:|----------------------:|-----------------------:|
|     mmlu     |                  35.32 |                  45.62 |                55.90 |                61.79 |                 67.96 |                 75.36 |                 77.24 |                  77.95 |
|    cmmlu     |                  31.55 |                  48.93 |                58.53 |                68.78 |                 75.07 |                 80.39 |                 82.48 |                  86.46 |
|  ceval-test  |                  36.88 |                  55.17 |                61.54 |                68.71 |                 74.80 |                 80.47 |                 81.53 |                  87.33 |
| GaokaoBench  |                  21.51 |                  46.19 |                59.11 |                70.55 |                 80.39 |                 86.15 |                 88.58 |                  89.59 |
|   triviaqa   |                  19.84 |                  35.81 |                48.93 |                53.65 |                 62.58 |                 74.72 |                 83.25 |                  86.20 |
|      nq      |                   7.42 |                  10.22 |                19.31 |                16.87 |                 20.53 |                 25.26 |                 35.21 |                  36.98 |
|  race-high   |                  49.03 |                  66.24 |                73.53 |                83.28 |                 87.51 |                 91.22 |                 91.11 |                  92.31 |
|  winogrande  |                  50.51 |                  51.07 |                57.54 |                65.27 |                 70.09 |                 77.90 |                 80.82 |                  82.32 |
|  hellaswag   |                  29.60 |                  41.71 |                60.45 |                71.58 |                 79.70 |                 88.56 |                 89.37 |                  91.11 |
|     bbh      |                  24.12 |                  26.82 |                43.15 |                38.12 |                 55.38 |                 69.28 |                 72.97 |                  71.04 |
|    gsm8k     |                   8.79 |                  27.60 |                47.61 |                56.25 |                 64.90 |                 79.91 |                 77.03 |                  79.53 |
|     math     |                   0.56 |                   4.94 |                 7.34 |                22.14 |                 32.22 |                 41.80 |                 45.22 |                  54.38 |
|  TheoremQA   |                   9.00 |                   9.25 |                13.88 |                12.25 |                 13.63 |                 19.25 |                 22.75 |                  17.50 |
|  humaneval   |                   9.15 |                  15.85 |                30.49 |                40.85 |                 50.00 |                 57.93 |                 60.37 |                  65.24 |
|     mbpp     |                  11.28 |                  22.57 |                43.58 |                50.58 |                 56.03 |                 65.37 |                 66.93 |                  68.48 |
|   lcbench    |                   0.00 |                   1.65 |                 5.56 |                 8.78 |                 14.42 |                 10.78 |                 18.77 |                  34.58 |
| GPQA_diamond |                  19.70 |                  29.80 |                25.25 |                31.82 |                 30.30 |                 31.31 |                 32.83 |                  35.86 |
|    IFEval    |                  13.12 |                  16.08 |                25.51 |                38.82 |                 42.51 |                 49.54 |                 51.02 |                  55.08 |

# Citation

```BibTeX
@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
```
