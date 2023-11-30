from mmengine.config import read_base

with read_base():
    from .models.qwen.hf_qwen_7b_chat import models
    from .datasets.collections.leaderboard.qwen_chat import datasets
    from .summarizers.leaderboard import summarizer

'''
dataset                                 version    metric            mode    qwen-7b-chat-hf
--------------------------------------  ---------  ----------------  ------  -----------------
--------- 考试 Exam ---------           -          -                 -       -
ceval                                   -          naive_average     gen     56.07
agieval                                 -          naive_average     mixed   39.51
mmlu                                    -          naive_average     gen     53.49
cmmlu                                   -          naive_average     gen     55.29
GaokaoBench                             -          weighted_average  gen     48.01
ARC-c                                   ca1e8e     accuracy          ppl     74.92
ARC-e                                   ca1e8e     accuracy          ppl     85.71
--------- 语言 Language ---------       -          -                 -       -
WiC                                     efbd01     accuracy          gen     51.41
chid-dev                                25f3d3     accuracy          ppl     77.72
afqmc-dev                               4a1636     accuracy          gen     69.00
WSC                                     678cb5     accuracy          ppl     67.31
tydiqa-goldp                            -          naive_average     gen     15.32
flores_100                              -          naive_average     gen     10.00
--------- 知识 Knowledge ---------      -          -                 -       -
BoolQ                                   463fee     accuracy          ppl     83.18
commonsense_qa                          ddaabf     accuracy          gen     76.41
triviaqa                                b6904f     score             gen     43.25
nq                                      23dc1a     score             gen     16.26
--------- 理解 Understanding ---------  -          -                 -       -
C3                                      e6778d     accuracy          gen     81.53
race-middle                             e0908b     accuracy          gen     83.01
race-high                               e0908b     accuracy          gen     77.79
openbookqa_fact                         49689a     accuracy          ppl     86.40
csl_dev                                 3c4211     accuracy          ppl     64.38
lcsts                                   0b3969     rouge1            gen     12.75
Xsum                                    207e69     rouge1            gen     20.21
eprstmt-dev                             ed0c5d     accuracy          ppl     85.00
lambada                                 de1af2     accuracy          gen     59.19
--------- 推理 Reasoning ---------      -          -                 -       -
cmnli                                   15e783     accuracy          ppl     48.08
ocnli                                   15e783     accuracy          ppl     51.40
AX_b                                    689df1     accuracy          ppl     65.67
AX_g                                    808a19     accuracy          ppl     76.12
RTE                                     808a19     accuracy          ppl     68.95
COPA                                    59f42c     accuracy          gen     92.00
ReCoRD                                  6f7cfc     score             gen     0.16
hellaswag                               8d79e0     accuracy          ppl     69.28
piqa                                    34eee7     accuracy          ppl     72.20
siqa                                    ea30d1     accuracy          ppl     72.88
math                                    2c0b9e     accuracy          gen     7.84
gsm8k                                   4c7f6e     accuracy          gen     45.41
drop                                    53a0a7     score             gen     39.62
openai_humaneval                        dd0dff     humaneval_pass@1  gen     10.98
mbpp                                    60ca11     score             gen     20.60
bbh                                     -          naive_average     gen     42.61
'''
