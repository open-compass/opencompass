# LVEval
## Introduction
The following introduction comes from the introduction in [LVEval](https://github.com/infinigence/LVEval)

```
LV-Eval是一个具备5个长度等级（16k、32k、64k、128k和256k）、最大文本测试长度达到256k的长文本评测基准。LV-Eval的平均文本长度达到102,380字，最小/最大文本长度为11,896/387,406字。LV-Eval主要有两类评测任务——单跳QA和多跳QA，共包含11个涵盖中英文的评测数据子集。LV-Eval设计时引入3个关键技术：干扰事实插入（Confusiong Facts Insertion，CFI）提高挑战性，关键词和短语替换（Keyword and Phrase Replacement，KPR）减少信息泄漏，以及基于关键词召回的评测指标（Answer Keywords，AK，指代结合答案关键词和字词黑名单的评价指标）提高评测数值客观性。我们希望LV-Eval为未来长文本大语言模型的研究发展提供有价值的性能参考。
LV-Eval is a challenging long-context benchmark with five length levels (16k, 32k, 64k, 128k, and 256k) reaching up to 256k words. The average number of words is 102,380, and the Min/Max number of words is 11,896/387,406. LV-Eval features two main tasks, single-hop QA and multi-hop QA, comprising 11 bilingual datasets. The design of LV-Eval has incorporated three key techniques, namely confusing facts insertion (CFI), keyword and phrase replacement (KPR), and keyword-recall-based metrics (AK, short for metics with Answer Keywords and word blacklist) design, which jointly provide a challenging, mitigated-knowledge-leakege, and more accurate evaluation of the long-context capability of LLMs. We anticipate that LV-Eval will serve as a valuable resource for supporting future research on long-context LLMs.
```

## Official link

### Paper

[_LV_-Eval: A Balanced Long-Context Benchmark with 5 Length Levels Up to 256K](https://arxiv.org/abs/2402.05136)

### Repository

[LVEval](https://github.com/infinigence/LVEval)

## Use cases

In evaluation scripts, add LVEval dataset as other datasets by using
```
from .datasets.lveval.lveval import LVEval_datasets as datasets
```

## Examples
Input example I (from lic_mixup datasets):
```
请根据下面给定的文章回答问题，问题和答案只与其中一篇文章有关。

文章：......文章 9\n\n标题：腐质酸\n内容：腐植酸是自然界中广泛存在的大分子有机物质，广泛应用于农林牧、石油、化工、建材、医药卫生、环保等各个领域。横跨几十个行业。特别是眼下提倡生态农业建设、无公害农业生产、绿色食品、无污染环保产品等，更使\"腐植酸\"备受推崇，事实证明，人类的生活和生存离不开腐植酸，它的确是一个发展中的有希望的朝阳产业，属于一个新型的特殊行业......

请现在基于上述文章回答下面的问题，问题和答案只与其中一篇文章有关。

问题：中国的文学受到印度哪些方面的影响？
回答：
```
Output example I (from chatglm3-6b-32k):
```
中国文学自印度文学大量吸收营养，在佛教东流之后，从语汇到修辞，从题材到体裁，即便审美取向也深受佛教与印度文学的感染。
```
Input example II (from factrecall_zh datasets):
```
请基于给定的文章回答下述问题。

文章：......庚子年间，贝多芬，乃一德裔美籍学士，研究于物理理学。彼其良图，探求相对论、量子力学，尤有大进。质能等价公式 E=mc²，千古独步，声名于当世。诺贝尔物理学奖、以资尊荣，兹矣荣耀之大典。论其学术，涉时空能量，影响深远，以其义非常人，广为当世所知，声名播于天下，实乃现代物理学之奠基者......

现在请基于上述文章回答下面的问题。

问题：被世人广泛推崇为现代物理学奠基人的科学家叫什么名字？
回答：
```
Output example II (from chatglm3-6b-32k):
```
贝多芬
```
## Evaluation results

```
dataset                                    version    metric         mode    bluelm-7b-chat-32k-hf
-----------------------------------------  ---------  -------------  ------  -----------------------
----------------------------------------   -          -              -       -
--------- LVEval All ---------             -          -              -       -
----------------------------------------   -          -              -       -
LVEval_qa                                  -          naive_average  gen     12.00
----------------------------------------   -          -              -       -
--------- LVEval Tasks All ---------       -          -              -       -
----------------------------------------   -          -              -       -
LVEval_single_hop_qa                       -          naive_average  gen     15.11
LVEval_single_hop_cqa                      -          naive_average  gen     9.21
LVEval_multi_hop_qa                        -          naive_average  gen     6.99
LVEval_multi_hop_cqa                       -          naive_average  gen     9.90
LVEval_factrecall_cqa                      -          naive_average  gen     21.28
----------------------------------------   -          -              -       -
--------- LVEval Datasets All ---------    -          -              -       -
----------------------------------------   -          -              -       -
LVEval_loogle_SD_mixup                     -          naive_average  gen     12.81
LVEval_cmrc_mixup                          -          naive_average  gen     17.41
LVEval_multifieldqa_en_mixup               -          naive_average  gen     7.10
LVEval_multifieldqa_zh_mixup               -          naive_average  gen     11.31
LVEval_dureader_mixup                      -          naive_average  gen     13.19
LVEval_loogle_CR_mixup                     -          naive_average  gen     5.17
LVEval_loogle_MIR_mixup                    -          naive_average  gen     2.60
LVEval_hotpotwikiqa_mixup                  -          naive_average  gen     10.20
LVEval_lic_mixup                           -          naive_average  gen     9.60
LVEval_factrecall_en                       -          naive_average  gen     23.67
LVEval_factrecall_zh                       -          naive_average  gen     18.90
----------------------------------------   -          -              -       -
--------- LVEval Single_Hop QA ---------   -          -              -       -
----------------------------------------   -          -              -       -
LVEval_loogle_SD_mixup_16k                 83bc25     LVEval_f1      gen     35.05
LVEval_loogle_SD_mixup_32k                 83bc25     LVEval_f1      gen     13.37
LVEval_loogle_SD_mixup_64k                 83bc25     LVEval_f1      gen     6.32
LVEval_loogle_SD_mixup_128k                83bc25     LVEval_f1      gen     5.28
LVEval_loogle_SD_mixup_256k                83bc25     LVEval_f1      gen     4.00
----------------------------------------   -          -              -       -
LVEval_cmrc_mixup_16k                      8bac4e     LVEval_f1      gen     46.45
LVEval_cmrc_mixup_32k                      8bac4e     LVEval_f1      gen     19.41
LVEval_cmrc_mixup_64k                      8bac4e     LVEval_f1      gen     11.10
LVEval_cmrc_mixup_128k                     8bac4e     LVEval_f1      gen     5.89
LVEval_cmrc_mixup_256k                     8bac4e     LVEval_f1      gen     4.22
----------------------------------------   -          -              -       -
--------- LVEval Single_Hop CQA ---------  -          -              -       -
----------------------------------------   -          -              -       -
LVEval_multifieldqa_en_mixup_16k           83bc25     LVEval_f1      gen     12.28
LVEval_multifieldqa_en_mixup_32k           83bc25     LVEval_f1      gen     4.64
LVEval_multifieldqa_en_mixup_64k           83bc25     LVEval_f1      gen     8.30
LVEval_multifieldqa_en_mixup_128k          83bc25     LVEval_f1      gen     5.63
LVEval_multifieldqa_en_mixup_256k          83bc25     LVEval_f1      gen     4.64
----------------------------------------   -          -              -       -
LVEval_multifieldqa_zh_mixup_16k           ac4a0d     LVEval_f1      gen     22.30
LVEval_multifieldqa_zh_mixup_32k           ac4a0d     LVEval_f1      gen     17.46
LVEval_multifieldqa_zh_mixup_64k           ac4a0d     LVEval_f1      gen     6.27
LVEval_multifieldqa_zh_mixup_128k          ac4a0d     LVEval_f1      gen     5.84
LVEval_multifieldqa_zh_mixup_256k          ac4a0d     LVEval_f1      gen     4.71
----------------------------------------   -          -              -       -
--------- LVEval Multi_Hop QA ---------    -          -              -       -
----------------------------------------   -          -              -       -
LVEval_dureader_mixup_16k                  8bac4e     LVEval_rouge   gen     18.04
LVEval_dureader_mixup_32k                  8bac4e     LVEval_rouge   gen     18.33
LVEval_dureader_mixup_64k                  8bac4e     LVEval_rouge   gen     12.56
LVEval_dureader_mixup_128k                 8bac4e     LVEval_rouge   gen     10.33
LVEval_dureader_mixup_256k                 8bac4e     LVEval_rouge   gen     6.69
----------------------------------------   -          -              -       -
LVEval_loogle_CR_mixup_16k                 83bc25     LVEval_f1      gen     9.35
LVEval_loogle_CR_mixup_32k                 83bc25     LVEval_f1      gen     7.42
LVEval_loogle_CR_mixup_64k                 83bc25     LVEval_f1      gen     3.18
LVEval_loogle_CR_mixup_128k                83bc25     LVEval_f1      gen     2.65
LVEval_loogle_CR_mixup_256k                83bc25     LVEval_f1      gen     3.27
----------------------------------------   -          -              -       -
LVEval_loogle_MIR_mixup_16k                83bc25     LVEval_f1      gen     4.50
LVEval_loogle_MIR_mixup_32k                83bc25     LVEval_f1      gen     3.19
LVEval_loogle_MIR_mixup_64k                83bc25     LVEval_f1      gen     2.34
LVEval_loogle_MIR_mixup_128k               83bc25     LVEval_f1      gen     1.76
LVEval_loogle_MIR_mixup_256k               83bc25     LVEval_f1      gen     1.20
----------------------------------------   -          -              -       -
--------- LVEval Multi_Hop CQA ---------   -          -              -       -
----------------------------------------   -          -              -       -
LVEval_hotpotwikiqa_mixup_16k              e3c368     LVEval_f1      gen     19.80
LVEval_hotpotwikiqa_mixup_32k              e3c368     LVEval_f1      gen     12.59
LVEval_hotpotwikiqa_mixup_64k              e3c368     LVEval_f1      gen     7.33
LVEval_hotpotwikiqa_mixup_128k             e3c368     LVEval_f1      gen     7.85
LVEval_hotpotwikiqa_mixup_256k             e3c368     LVEval_f1      gen     3.42
----------------------------------------   -          -              -       -
LVEval_lic_mixup_16k                       fdd540     LVEval_f1      gen     21.36
LVEval_lic_mixup_32k                       fdd540     LVEval_f1      gen     12.92
LVEval_lic_mixup_64k                       fdd540     LVEval_f1      gen     4.62
LVEval_lic_mixup_128k                      fdd540     LVEval_f1      gen     4.25
LVEval_lic_mixup_256k                      fdd540     LVEval_f1      gen     4.85
----------------------------------------   -          -              -       -
--------- LVEval Factrecall CQA ---------  -          -              -       -
----------------------------------------   -          -              -       -
LVEval_factrecall_en_16k                   fba966     f1             gen     58.33
LVEval_factrecall_en_32k                   fba966     f1             gen     32.17
LVEval_factrecall_en_64k                   fba966     f1             gen     15.33
LVEval_factrecall_en_128k                  fba966     f1             gen     8.50
LVEval_factrecall_en_256k                  fba966     f1             gen     4.00
----------------------------------------   -          -              -       -
LVEval_factrecall_zh_16k                   ef3320     f1             gen     20.00
LVEval_factrecall_zh_32k                   ef3320     f1             gen     38.00
LVEval_factrecall_zh_64k                   ef3320     f1             gen     20.50
LVEval_factrecall_zh_128k                  ef3320     f1             gen     11.00
LVEval_factrecall_zh_256k                  ef3320     f1             gen     5.00
```
