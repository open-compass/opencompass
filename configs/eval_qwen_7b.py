from mmengine.config import read_base

with read_base():
    from .models.qwen.hf_qwen_7b import models
    from .datasets.collections.leaderboard.qwen import datasets
    from .summarizers.leaderboard import summarizer

'''
dataset                                 version    metric            mode    qwen-7b-hf
--------------------------------------  ---------  ----------------  ------  ------------
--------- 考试 Exam ---------           -          -                 -       -
ceval                                   -          naive_average     ppl     58.65
agieval                                 -          naive_average     mixed   40.49
mmlu                                    -          naive_average     ppl     57.78
cmmlu                                   -          naive_average     ppl     58.57
GaokaoBench                             -          weighted_average  mixed   51.76
ARC-c                                   72cf91     accuracy          gen     83.73
ARC-e                                   72cf91     accuracy          gen     90.65
--------- 语言 Language ---------       -          -                 -       -
WiC                                     ce62e6     accuracy          ppl     51.10
chid-dev                                25f3d3     accuracy          ppl     86.63
afqmc-dev                               cc328c     accuracy          ppl     69.00
WSC                                     678cb5     accuracy          ppl     63.46
tydiqa-goldp                            -          naive_average     gen     19.98
flores_100                              -          naive_average     gen     3.20
--------- 知识 Knowledge ---------      -          -                 -       -
BoolQ                                   463fee     accuracy          ppl     83.00
commonsense_qa                          0d8e25     accuracy          ppl     67.49
triviaqa                                b6904f     score             gen     40.45
nq                                      b6904f     score             gen     14.16
--------- 理解 Understanding ---------  -          -                 -       -
C3                                      e6778d     accuracy          gen     75.29
race-middle                             73bdec     accuracy          ppl     90.53
race-high                               73bdec     accuracy          ppl     87.71
openbookqa_fact                         fa871c     accuracy          gen     92.20
csl_dev                                 3c4211     accuracy          ppl     56.25
lcsts                                   0b3969     rouge1            gen     12.38
Xsum                                    207e69     rouge1            gen     36.00
eprstmt-dev                             101429     accuracy          gen     89.38
lambada                                 de1af2     accuracy          gen     67.88
--------- 推理 Reasoning ---------      -          -                 -       -
cmnli                                   15e783     accuracy          ppl     54.85
ocnli                                   1471e7     accuracy          gen     42.34
AX_b                                    793c72     accuracy          gen     58.61
AX_g                                    c4c886     accuracy          gen     69.10
RTE                                     c4c886     accuracy          gen     57.76
COPA                                    59f42c     accuracy          gen     88.00
ReCoRD                                  3e0689     score             gen     27.78
hellaswag                               06a1e2     accuracy          gen     92.47
piqa                                    24369d     accuracy          gen     78.02
siqa                                    ea30d1     accuracy          ppl     75.03
math                                    2c0b9e     accuracy          gen     11.06
gsm8k                                   4c7f6e     accuracy          gen     50.87
drop                                    53a0a7     score             gen     44.95
openai_humaneval                        dd0dff     humaneval_pass@1  gen     23.78
mbpp                                    60ca11     score             gen     31.20
bbh                                     -          naive_average     gen     40.03
'''
