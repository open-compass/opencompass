# CompassArena-SubjectiveBench (Pairwise Eval with Bradley-Terry Model)

## Introduction

The following introduction comes from the abstract of [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132):

>Large Language Models (LLMs) have unlocked new capabilities and applications; however, evaluating the alignment with human preferences still poses significant challenges. To address this issue, we introduce Chatbot Arena, an open platform for evaluating LLMs based on human preferences. Our methodology employs a pairwise comparison approach and leverages input from a diverse user base through crowdsourcing. The platform has been operational for several months, amassing over 240K votes. This paper describes the platform, analyzes the data we have collected so far, and explains the tried-and-true statistical methods we are using for efficient and accurate evaluation and ranking of models. We confirm that the crowdsourced questions are sufficiently diverse and discriminating and that the crowdsourced human votes are in good agreement with those of expert raters. These analyses collectively establish a robust foundation for the credibility of Chatbot Arena. Because of its unique value and openness, Chatbot Arena has emerged as one of the most referenced LLM leaderboards, widely cited by leading LLM developers and companies.

For this dataset, we adapt the Bradley-Terry rating system from FastChat to the subjective evaluation setting, but replacing human evaluators with LLM-as-a-judge.


## Official Links

- Paper: [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132)
- GitHub Repository: [FastChat](https://github.com/lm-sys/FastChat/tree/main)


## Overview and Usage

### Inference

During the inference stage, each LLM makes an inference based on the question presented (single question for single turn and an entire conversation for multi-turn).

### Evaluation

During the evaluation stage, the judge model respond with a critique and chooses the LLM with a better answer for each pair. This preference will be used later to form the "winner" response variable in the postprocessor. Note that the predictions for each model must be saved (by setting `keep_predictions=True` in the evaluator config) in order for the postporcessor to calculate style features. See this [example](`opencompass/configs/datasets/subjective/compass_arena_subjective_bench/singleturn/pairwise_bt_judge.py`) for more details.


#### Postprocessor
After evaluation by the judge model, we gather the pairwise matchups and any additional group variables (e.g. difficulty, category) in the postprocessor. Note that the LLM predictions ("prediction1" and "prediction2") must be passed on from the inference stage, otherwise, an error will be thrown.


### Summary

After inference by the judge model in the evaluation stage, we fit a Bradley-Terry model (statistical model) in order to estimate the rating and ranking of each LLM with an option to include style features and control variables on groups. The settings below control specification of the BT model as well as how results are being reported:

- `rating_system`: The rating system used. Currently only supports "bradleyterry".

- `num_bootstrap`: The number of bootstraps for estimating the confidence intervals of ratings.

- `with_control_vars`: Whether to include additional covariates (including style features and group variables) when fitting the BT model.

- `normalize_style_features`: Whether to normalize style features BEFORE fitting the BT model (implementation by FastChat). Turn this off for easier interpretation of odds ratios (when `odds_ratio==True`).

- `odds_ratio`: Whether to report odds ratios ($e^{\beta_i}$) instead of the original coefficients. See section "Estimated Coefficients of Control variables" for more explanation.

- `groups`: List of group variables to include while fitting the BT model. These must be available in the input dataset for each observation. Group variables are assumed to be categorical and one-hot encoding is automatically performed before model fitting.


### Config Files

1. Dataset configs:

    - single turn: `opencompass/configs/datasets/subjective/compass_arena_subjective_bench/singleturn/pairwise_bt_judge.py`
    - multi-turn: `opencompass/configs/datasets/subjective/compass_arena_subjective_bench/multiturn/pairwise_bt_judge.py`

2. Evaluation config:

    - `examples/eval_compassarena_subjectivebench_bradleyterry.py`

## Evaluation Results

### Bradley-Terry Rating

The rating of each model is a scaled version of the estimated "strength" coefficients of the fitted Bradley-Terry model. We use the Elo scale with an initial rating of 1000 and a scaling factor of 400 to match the scale used in [CompassArena](https://opencompass.org.cn/arena). Furthermore, we anchor the ratings on the base model as it naturally represents the reference model we are comparing against. This is why the base model always have a rating of 1000 with a zero standard deviation.

```
      dataset version             base_model     metric     mode  ranking  ranking_ub                      model_name   rating  rating_q975  rating_q025  std_dev  num_battles
0  singleturn  635142  Qwen-2.5-72B-Instruct  bt_rating      gen        1           1           Qwen-2.5-72B-Instruct  1000.00      1000.00      1000.00     0.00         4229
1  singleturn  635142  Qwen-2.5-72B-Instruct  bt_rating      gen        2           2  qwen2.5-32b-instruct-turbomind   926.54       941.72       908.29     8.21         1055
2  singleturn  635142  Qwen-2.5-72B-Instruct  bt_rating      gen        3           2  qwen2.5-14b-instruct-turbomind   907.23       921.08       897.09     6.68         1055
3  singleturn  635142  Qwen-2.5-72B-Instruct  bt_rating      gen        4           2     qwen2-7b-instruct-turbomind   901.99       919.06       885.95     8.44         1060
4  singleturn  635142  Qwen-2.5-72B-Instruct  bt_rating      gen        5           2   qwen2.5-7b-instruct-turbomind   893.03       910.58       877.02     8.65         1059
5   multiturn  fff2b4  Qwen-2.5-72B-Instruct  bt_rating  unknown        1           1           Qwen-2.5-72B-Instruct  1000.00      1000.00      1000.00     0.00         1127
6   multiturn  fff2b4  Qwen-2.5-72B-Instruct  bt_rating  unknown        2           2  qwen2.5-32b-instruct-turbomind   942.53       972.14       903.84    18.89          282
7   multiturn  fff2b4  Qwen-2.5-72B-Instruct  bt_rating  unknown        3           2     qwen2-7b-instruct-turbomind   940.34       974.22       895.80    21.72          282
8   multiturn  fff2b4  Qwen-2.5-72B-Instruct  bt_rating  unknown        4           2  qwen2.5-14b-instruct-turbomind   929.09       959.98       896.80    18.16          282
9   multiturn  fff2b4  Qwen-2.5-72B-Instruct  bt_rating  unknown        5           2   qwen2.5-7b-instruct-turbomind   907.07       936.71       876.88    16.87          281
```

### Estimated Coefficients of Control variables

The scale and interpretation of these numbers depend on the summarizer settings for `CompassArenaBradleyTerrySummarizer`. If `normalize_style_features` is set, the style features are the normalized relative difference between model A and B, with the following form:
$$
\text{normalize }\left(\frac{\text{feature}_A - \text{feature}_B}{\text{feature}_A + \text{feature}_B}\right)
$$

See [Does Style Matter?](https://blog.lmarena.ai/blog/2024/style-control/) for more information.

Additionally, if `odds_ratio` is set, the odds ratios are returned instead of the raw coefficients. In other words, we report:

$$
\text{OddsRatio}_i = \frac{e^{\beta_0 + \beta_i(x_i+1) + \sum_{j\ne i}^m\beta_jx_j}}{e^{\beta_0 + \beta_ix_i + \sum_{j\ne i}^m\beta_jx_j}} = e^{\beta_i}
$$

which can be interpretted as the multiplicative increase in odds for every 1-unit increase in $x_i$.

For example, the following results are reported with `normalize_style_features==False` and `odds_ratio==True`:
```
{
    "singleturn": {
        "Qwen-2.5-72B-Instruct": {
            "sum_assistant_tokens": 6.577376545800252,
            "header_count": 1.4880636137846999,
            "list_count": 1.1558594451186806,
            "bold_count": 1.7918326386585717,
            "difficulty_Advanced": 1.0281620474711213,
            "difficulty_Easy": 1.0557367496235666,
            "difficulty_Medium": 1.1768581931447049,
            "category_人类对齐": 0.8087074923883157,
            "category_代码": 1.2717334332407775,
            "category_创作": 1.0430652013278148,
            "category_推理": 1.1592759054335746,
            "category_日常对话": 0.979047716903164,
            "category_自然语言处理": 1.006707704304149,
            "category_角色扮演": 1.2296103927210726,
            "category_重写": 0.7952522120597192,
            "category_领域知识问答": 1.0658003517547319
        }
    },
    "multiturn": {
        "Qwen-2.5-72B-Instruct": {
            "sum_assistant_tokens": 4.470153434554273,
            "header_count": 1.130542616688942,
            "list_count": 1.4753419673439991,
            "bold_count": 1.476348454534956,
            "difficulty_Advanced": 1.1668553174437737,
            "difficulty_Easy": 1.142118410006132,
            "difficulty_Medium": 0.9651479035385795,
            "category_人类对齐": 0.9606676068409767,
            "category_代码": 0.9348722519214725,
            "category_创作": 1.0362490715530026,
            "category_推理": 0.8546385641566406,
            "category_日常对话": 1.0481269627721679,
            "category_自然语言处理": 1.358391853082614,
            "category_角色扮演": 1.0432636535119493,
            "category_重写": 0.7398232857603452,
            "category_领域知识问答": 1.4715970942932421
        }
    }
}
```
Example Interpretation:
- For the single turn dataset with "Qwen-2.5-72B-Instruct" as the base model, if all else stay constant, the odds of winning is 6.6 times greater for every unit increase in the relative difference (unnormalized) in response length between model A and B.

- For the multi-turn dataset with "Qwen-2.5-72B-Instruct" as the base model, if all else stay constant, the odds of winning is 26% smaller (1-0.74) for "rewrite" (重写) category questions compared to non-rewrite questions.


## Citation
```
@misc{chiang2024chatbotarenaopenplatform,
      title={Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference},
      author={Wei-Lin Chiang and Lianmin Zheng and Ying Sheng and Anastasios Nikolas Angelopoulos and Tianle Li and Dacheng Li and Hao Zhang and Banghua Zhu and Michael Jordan and Joseph E. Gonzalez and Ion Stoica},
      year={2024},
      eprint={2403.04132},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2403.04132},
}

@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
