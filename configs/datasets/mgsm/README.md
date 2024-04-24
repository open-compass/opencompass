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
Answer:18
```


## Evaluation results

```
dataset    version    metric    mode      internlm2-chat
---------  ---------  --------  ------  ----------------
mgsm_en    5ddc62     Acc       gen                    0
mgsm_zh    5ddc62     Acc       gen                    0
mgsm_bn    5ddc62     Acc       gen                    0
mgsm_de    5ddc62     Acc       gen                   10
mgsm_es    5ddc62     Acc       gen                   10
mgsm_fr    5ddc62     Acc       gen                    0
mgsm_ja    5ddc62     Acc       gen                    0
mgsm_ru    5ddc62     Acc       gen                   10
mgsm_sw    5ddc62     Acc       gen                   10
mgsm_te    5ddc62     Acc       gen                    0
mgsm_th    5ddc62     Acc       gen                    0
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
