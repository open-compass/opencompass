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

- `groups`: List of group variables to include while fitting the BT model. These must be available in the input dataset for each observation. Group variables are assumed to be categorical and one-hot encoding is automatically performed with the first category removed before model fitting.


### Config Files

1. Dataset configs:

    - single turn: `opencompass/configs/datasets/subjective/compass_arena_subjective_bench/singleturn/pairwise_bt_judge.py`
    - multi-turn: `opencompass/configs/datasets/subjective/compass_arena_subjective_bench/multiturn/pairwise_bt_judge.py`

2. Evaluation config:

    - `configs/eval_compassarena_subjectivebench_bradleyterry.py`

## Evaluation Results

### Bradley-Terry Rating

The rating of each model is a scaled version of the estimated "strength" coefficients of the fitted Bradley-Terry model. We use the Elo scale with an initial rating of 1000 and a scaling factor of 400 to match the scale used in [CompassArena](https://opencompass.org.cn/arena). Furthermore, we anchor the ratings on the base model as it naturally represents the reference model we are comparing against. This is why the base model always have a rating of 1000 with a zero standard deviation.

```
dataset	version	base_model	metric	mode	ranking	ranking_ub	model_name	rating	rating_q975	rating_q025	std_dev	num_battles
singleturn	635142	Qwen-2.5-72B-Instruct	bt_rating	gen	1	1	Qwen-2.5-72B-Instruct	1000	1000	1000	0	4229
singleturn	635142	Qwen-2.5-72B-Instruct	bt_rating	gen	2	2	qwen2.5-32b-instruct-turbomind	926.2231474	945.0262849	908.2263054	9.019177299	1055
singleturn	635142	Qwen-2.5-72B-Instruct	bt_rating	gen	3	2	qwen2.5-14b-instruct-turbomind	906.9859382	925.4317943	889.7925857	9.168838569	1055
singleturn	635142	Qwen-2.5-72B-Instruct	bt_rating	gen	4	2	qwen2-7b-instruct-turbomind	901.7568797	918.0514477	882.8266488	9.270810688	1060
singleturn	635142	Qwen-2.5-72B-Instruct	bt_rating	gen	5	3	qwen2.5-7b-instruct-turbomind	892.7376763	904.967507	879.0671151	7.582809959	1059
multiturn	fff2b4	Qwen-2.5-72B-Instruct	bt_rating	unknown	1	1	Qwen-2.5-72B-Instruct	1000	1000	1000	0	1127
multiturn	fff2b4	Qwen-2.5-72B-Instruct	bt_rating	unknown	2	2	qwen2.5-32b-instruct-turbomind	942.1016777	976.945207	907.3027472	19.27814565	282
multiturn	fff2b4	Qwen-2.5-72B-Instruct	bt_rating	unknown	3	2	qwen2-7b-instruct-turbomind	940.0731684	970.5585368	892.3181852	22.04105572	282
multiturn	fff2b4	Qwen-2.5-72B-Instruct	bt_rating	unknown	4	2	qwen2.5-14b-instruct-turbomind	928.7889118	964.8279798	892.5656754	19.08905489	282
multiturn	fff2b4	Qwen-2.5-72B-Instruct	bt_rating	unknown	5	2	qwen2.5-7b-instruct-turbomind	906.7686895	932.9998533	874.5200202	17.2591473	281
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
            "sum_assistant_tokens": 6.53720403351394,
            "header_count": 1.4792109070935404,
            "list_count": 1.160034285851486,
            "bold_count": 1.7895403566484283,
            "difficulty_Easy": 0.9759045078292496,
            "difficulty_Medium": 1.0871822638256954,
            "category_代码": 1.354417670414851,
            "category_创作": 1.1094959274269027,
            "category_推理": 1.2357933187526577,
            "category_日常对话": 1.0410875930674186,
            "category_自然语言处理": 1.0709623294409785,
            "category_角色扮演": 1.309683380779042,
            "category_重写": 0.8467240295931602,
            "category_领域知识问答": 1.1352276271498163
        }
    },
    "multiturn": {
        "Qwen-2.5-72B-Instruct": {
            "sum_assistant_tokens": 4.4598969821940715,
            "header_count": 1.1290504434335653,
            "list_count": 1.469607623465334,
            "bold_count": 1.4738629875792915,
            "difficulty_Easy": 1.0078742450843936,
            "difficulty_Medium": 0.8526743244695739,
            "category_代码": 1.0669382695680414,
            "category_创作": 1.1813395478003295,
            "category_推理": 0.9752426633584584,
            "category_日常对话": 1.1967182467429063,
            "category_自然语言处理": 1.5484033674244435,
            "category_角色扮演": 1.190025282220524,
            "category_重写": 0.8443945949829733,
            "category_领域知识问答": 1.6758254499585854
        }
    }
}
```
Example Interpretation:
- For the single turn dataset with "Qwen-2.5-72B-Instruct" as the base model, if all else stay constant, the odds of winning is 6.5 times greater for every unit increase in the relative difference (unnormalized) in response length between model A and B.

- For the multi-turn dataset with "Qwen-2.5-72B-Instruct" as the base model, if all else stay constant, the odds of winning is 16% smaller (1-0.84) for "rewrite" (重写) category questions compared to non-rewrite questions.


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
