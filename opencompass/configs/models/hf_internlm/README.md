# InternLM Model Details

## InternLM

InternLM is pre-trained on a large corpora with 1.6T tokens with a multi-phase progressive process, and then fine-tuned to align with human preferences. We also developed a training system called Uniscale-LLM for efficient large language model training. The evaluation on a number of benchmarks shows that InternLM achieves state-of-the-art performance in multiple aspects, including knowledge understanding, reading comprehension, mathematics, and coding. With such well-rounded capabilities, InternLM achieves outstanding performances on comprehensive exams, including MMLU, AGIEval, C-Eval and GAOKAO-Bench, without resorting to external tools. On these benchmarks, InternLM not only significantly outperforms open-source models, but also obtains superior performance compared to ChatGPT. Also, InternLM demonstrates excellent capability of understanding Chinese language and Chinese culture, which makes it a suitable foundation model to support Chinese-oriented language applications.

## InternLM2

The evolution of Large Language Models (LLMs) like ChatGPT and GPT-4 has sparked discussions on the advent of Artificial General Intelligence (AGI). However, replicating such advancements in open-source models has been challenging. This paper introduces InternLM2, an open-source LLM that outperforms its predecessors in comprehensive evaluations across 6 dimensions and 30 benchmarks, long-context modeling, and open-ended subjective evaluations through innovative pre-training and optimization techniques. The pre-training process of InternLM2 is meticulously detailed, highlighting the preparation of diverse data types including text, code, and long-context data. InternLM2 efficiently captures long-term dependencies, initially trained on 4k tokens before advancing to 32k tokens in pre-training and fine-tuning stages, exhibiting remarkable performance on the 200k "Needle-in-a-Haystack" test. InternLM2 is further aligned using Supervised Fine-Tuning (SFT) and a novel Conditional Online Reinforcement Learning from Human Feedback (COOL RLHF) strategy that addresses conflicting human preferences and reward hacking. By releasing InternLM2 models in different training stages and model sizes, we provide the community with insights into the model's evolution.

# Evaluation Command

## Base Models

```bash
python3 run.py --models hf_internlm2_7b --datasets mmlu_ppl_ac766d --debug
python3 run.py --models hf_internlm2_7b --datasets cmmlu_ppl_041cbf --debug
python3 run.py --models hf_internlm2_7b --datasets ceval_internal_ppl_93e5ce --debug
python3 run.py --models hf_internlm2_7b --datasets GaokaoBench_no_subjective_gen_d21e37 --debug
python3 run.py --models hf_internlm2_7b --datasets triviaqa_wiki_1shot_gen_20a989 --debug
python3 run.py --models hf_internlm2_7b --datasets nq_open_1shot_gen_20a989 --debug
python3 run.py --models hf_internlm2_7b --datasets race_ppl_abed12 --debug
python3 run.py --models hf_internlm2_7b --datasets winogrande_5shot_ll_252f01 --debug
python3 run.py --models hf_internlm2_7b --datasets hellaswag_10shot_ppl_59c85e --debug
python3 run.py --models hf_internlm2_7b --datasets bbh_gen_98fba6 --debug
python3 run.py --models hf_internlm2_7b --datasets gsm8k_gen_17d0dc --debug
python3 run.py --models hf_internlm2_7b --datasets math_4shot_base_gen_db136b --debug
python3 run.py --models hf_internlm2_7b --datasets TheoremQA_5shot_gen_6f0af8 --debug
python3 run.py --models hf_internlm2_7b --datasets deprecated_humaneval_gen_d2537e --debug
python3 run.py --models hf_internlm2_7b --datasets sanitized_mbpp_gen_742f0c --debug
python3 run.py --models hf_internlm2_7b --datasets lcbench_gen_5ff288 --debug
python3 run.py --models hf_internlm2_7b --datasets gpqa_ppl_6bf57a --debug
```

## Chat Models

```bash
python3 run.py --models hf_internlm2_chat_7b --datasets mmlu_gen_4d595a --debug
python3 run.py --models hf_internlm2_chat_7b --datasets cmmlu_gen_c13365 --debug
python3 run.py --models hf_internlm2_chat_7b --datasets ceval_internal_gen_2daf24 --debug
python3 run.py --models hf_internlm2_chat_7b --datasets GaokaoBench_no_subjective_gen_4c31db --debug
python3 run.py --models hf_internlm2_chat_7b --datasets triviaqa_wiki_1shot_gen_eaf81e --debug
python3 run.py --models hf_internlm2_chat_7b --datasets nq_open_1shot_gen_01cf41 --debug
python3 run.py --models hf_internlm2_chat_7b --datasets race_gen_69ee4f --debug
python3 run.py --models hf_internlm2_chat_7b --datasets winogrande_5shot_gen_b36770 --debug
python3 run.py --models hf_internlm2_chat_7b --datasets hellaswag_10shot_gen_e42710 --debug
python3 run.py --models hf_internlm2_chat_7b --datasets bbh_gen_5b92b0 --debug
python3 run.py --models hf_internlm2_chat_7b --datasets gsm8k_gen_1d7fe4 --debug
python3 run.py --models hf_internlm2_chat_7b --datasets math_0shot_gen_393424 --debug
python3 run.py --models hf_internlm2_chat_7b --datasets TheoremQA_5shot_gen_6f0af8 --debug
python3 run.py --models hf_internlm2_chat_7b --datasets humaneval_gen_8e312c --debug
python3 run.py --models hf_internlm2_chat_7b --datasets sanitized_mbpp_mdblock_gen_a447ff --debug
python3 run.py --models hf_internlm2_chat_7b --datasets lcbench_gen_5ff288 --debug
python3 run.py --models hf_internlm2_chat_7b --datasets gpqa_gen_4baadb --debug
python3 run.py --models hf_internlm2_chat_7b --datasets IFEval_gen_3321a3 --debug
```

# Benchmarks

We provide reference results for the classifical models, you can reproduce these results by following the aforementioned commands.

## Base Models

|   dataset    |   internlm2-1.8b-turbomind |   internlm2-7b-turbomind |   internlm2-20b-turbomind |
|:------------:|---------------------------:|-------------------------:|--------------------------:|
|     mmlu     |                      45.99 |                    65.84 |                     67.58 |
|    cmmlu     |                      45.27 |                    66.17 |                     68.28 |
|  ceval-test  |                      44.79 |                    63.54 |                     67.28 |
| GaokaoBench  |                      23.78 |                    41.41 |                     58.99 |
|   triviaqa   |                      38.42 |                    69.15 |                     74.03 |
|      nq      |                      20.66 |                    41.05 |                     43.55 |
|  race-high   |                      64.72 |                    72.56 |                     72.90 |
|  winogrande  |                      66.77 |                    83.50 |                     84.69 |
|  hellaswag   |                      44.86 |                    89.52 |                     91.41 |
|     bbh      |                      36.03 |                    63.56 |                     71.29 |
|    gsm8k     |                      30.40 |                    69.98 |                     76.80 |
|     math     |                       9.42 |                    25.16 |                     32.24 |
|  TheoremQA   |                      10.50 |                    21.88 |                     26.00 |
|  humaneval   |                      30.49 |                    48.17 |                     51.83 |
|     mbpp     |                      30.74 |                    54.47 |                     59.92 |
|   lcbench    |                       4.34 |                    12.16 |                     18.46 |
| GPQA_diamond |                      24.24 |                    28.28 |                     31.31 |

## Chat Models

|   dataset    |   internlm2-chat-1.8b-hf |   internlm2-chat-1.8b-sft-hf |   internlm2-chat-7b-hf |   internlm2-chat-7b-sft-hf |   internlm2-chat-20b-hf |   internlm2-chat-20b-sft-hf |
|:------------:|-------------------------:|-----------------------------:|-----------------------:|---------------------------:|------------------------:|----------------------------:|
|     mmlu     |                    47.58 |                        47.44 |                  63.05 |                      63.33 |                   67.37 |                       67.34 |
|    cmmlu     |                    46.11 |                        46.27 |                  62.10 |                      62.38 |                   66.26 |                       66.39 |
|  ceval-test  |                    47.04 |                        47.19 |                  58.75 |                      58.96 |                   63.12 |                       63.16 |
| GaokaoBench  |                    29.73 |                        28.79 |                  54.54 |                      55.39 |                   57.95 |                       57.62 |
|   triviaqa   |                    46.69 |                        46.50 |                  69.54 |                      70.75 |                   75.53 |                       75.90 |
|      nq      |                    19.09 |                        18.14 |                  28.73 |                      30.78 |                   28.75 |                       34.10 |
|  race-high   |                    73.87 |                        73.81 |                  84.51 |                      84.88 |                   88.02 |                       88.11 |
|  winogrande  |                    57.62 |                        57.93 |                  73.56 |                      73.80 |                   81.06 |                       81.37 |
|  hellaswag   |                    60.47 |                        61.58 |                  84.80 |                      85.21 |                   88.48 |                       88.95 |
|     bbh      |                    37.69 |                        37.12 |                  57.83 |                      57.19 |                   68.24 |                       69.38 |
|    gsm8k     |                    39.73 |                        36.85 |                  69.90 |                      69.83 |                   75.21 |                       76.95 |
|     math     |                    14.06 |                        13.10 |                  28.08 |                      27.60 |                   34.68 |                       32.54 |
|  TheoremQA   |                    13.63 |                        12.88 |                  18.50 |                      18.75 |                   23.00 |                       25.12 |
|  humaneval   |                    33.54 |                        34.15 |                  56.71 |                      61.59 |                   67.68 |                       67.68 |
|     mbpp     |                    39.69 |                        36.19 |                  57.59 |                      55.64 |                   68.87 |                       69.65 |
|   lcbench    |                     4.52 |                         3.56 |                  14.60 |                      14.34 |                   19.64 |                       20.55 |
| GPQA_diamond |                    25.76 |                        26.26 |                  28.28 |                      27.27 |                   30.30 |                       29.29 |
|    IFEval    |                    18.30 |                        18.67 |                  34.75 |                      39.19 |                   36.41 |                       44.55 |

# Citation

```BibTeX
@misc{2023internlm,
    title={InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities},
    author={InternLM Team},
    howpublished = {\url{https://github.com/InternLM/InternLM-techreport}},
    year={2023}
}
@misc{cai2024internlm2,
      title={InternLM2 Technical Report},
      author={Zheng Cai and Maosong Cao and Haojiong Chen and Kai Chen and Keyu Chen and Xin Chen and Xun Chen and Zehui Chen and Zhi Chen and Pei Chu and Xiaoyi Dong and Haodong Duan and Qi Fan and Zhaoye Fei and Yang Gao and Jiaye Ge and Chenya Gu and Yuzhe Gu and Tao Gui and Aijia Guo and Qipeng Guo and Conghui He and Yingfan Hu and Ting Huang and Tao Jiang and Penglong Jiao and Zhenjiang Jin and Zhikai Lei and Jiaxing Li and Jingwen Li and Linyang Li and Shuaibin Li and Wei Li and Yining Li and Hongwei Liu and Jiangning Liu and Jiawei Hong and Kaiwen Liu and Kuikun Liu and Xiaoran Liu and Chengqi Lv and Haijun Lv and Kai Lv and Li Ma and Runyuan Ma and Zerun Ma and Wenchang Ning and Linke Ouyang and Jiantao Qiu and Yuan Qu and Fukai Shang and Yunfan Shao and Demin Song and Zifan Song and Zhihao Sui and Peng Sun and Yu Sun and Huanze Tang and Bin Wang and Guoteng Wang and Jiaqi Wang and Jiayu Wang and Rui Wang and Yudong Wang and Ziyi Wang and Xingjian Wei and Qizhen Weng and Fan Wu and Yingtong Xiong and Chao Xu and Ruiliang Xu and Hang Yan and Yirong Yan and Xiaogui Yang and Haochen Ye and Huaiyuan Ying and Jia Yu and Jing Yu and Yuhang Zang and Chuyu Zhang and Li Zhang and Pan Zhang and Peng Zhang and Ruijie Zhang and Shuo Zhang and Songyang Zhang and Wenjian Zhang and Wenwei Zhang and Xingcheng Zhang and Xinyue Zhang and Hui Zhao and Qian Zhao and Xiaomeng Zhao and Fengzhe Zhou and Zaida Zhou and Jingming Zhuo and Yicheng Zou and Xipeng Qiu and Yu Qiao and Dahua Lin},
      year={2024},
      eprint={2403.17297},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
