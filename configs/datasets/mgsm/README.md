# MGSM

## Introduction

The following introduction comes from the abstract in [Language models are multilingual chain-of-thought reasoners](https://arxiv.org/abs/2210.03057)

```
We introduce the Multilingual Grade School Math (MGSM) benchmark, by manually translating 250 grade-school math problems from the GSM8K dataset into ten typologically diverse languages.
```

## Official link

### Paper

[Language models are multilingual chain-of-thought reasoners](https://arxiv.org/abs/2210.03057)

### Repository

[MGSM](https://github.com/google-research/url-nlp)

## Examples

Input example I:

```
Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
```

Output example I (from GPT-4):

```
Answer: 18
```

## Evaluation results

```
dataset         version    metric         mode      llama-3-8b-instruct-hf
--------------  ---------  -------------  ------  ------------------------
mgsm_bn         b65151     accuracy       gen                        14.4
mgsm_de         2cc8ae     accuracy       gen                        60
mgsm_en         5de71e     accuracy       gen                        76
mgsm_es         d6b459     accuracy       gen                        61.6
mgsm_fr         813e3c     accuracy       gen                        54.4
mgsm_ja         04424f     accuracy       gen                        42.8
mgsm_ru         400469     accuracy       gen                        62.8
mgsm_sw         9e41ed     accuracy       gen                         0.8
mgsm_te         346d97     accuracy       gen                         0
mgsm_th         e70bee     accuracy       gen                        44
mgsm_zh         d5cf30     accuracy       gen                        28.4
mgsm_latin      -          naive_average  gen                        50.56
mgsm_non_latin  -          naive_average  gen                        32.07
mgsm            -          naive_average  gen                        40.47
```

## Reference

```
@article{shi2022language,
  title={Language models are multilingual chain-of-thought reasoners},
  author={Shi, Freda and Suzgun, Mirac and Freitag, Markus and Wang, Xuezhi and Srivats, Suraj and Vosoughi, Soroush and Chung, Hyung Won and Tay, Yi and Ruder, Sebastian and Zhou, Denny and others},
  journal={arXiv preprint arXiv:2210.03057},
  year={2022}
}
```
