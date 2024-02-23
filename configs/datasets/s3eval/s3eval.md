# S3Eval
## Introduction
The following introduction comes from the abstract in [S3Eval: A Synthetic, Scalable and Systematic Evaluation Suite for Large Language Models](arxiv.org/abs/2310.15147)

S3Eval, our latest contribution to the field, addresses the critical need for comprehensive evaluation resources for Large Language Models (LLMs). In the pursuit of understanding **long-context comprehension** and enhancing **reasoning capabilities**, we present a benchmarking suite that is both synthetic and scalable.

Operating on SQL execution tasks, S3Eval challenges LLMs with randomly generated tables and SQL queries, evaluating their ability to produce accurate execution results. This benchmark stands out for its versatility and scalability, providing **unlimited evaluation resources** for a robust assessment of LLM capabilities.

In this latest submission, we have generated a batch of high-quality data, encompassing nearly all types of queries with strong diversity. Moreover, the length of the tables **spans from 200 to 200K**, enabling a systematic evaluation of the long-context capabilities of the models.

For researchers and practitioners alike, S3Eval holds the promise of uncovering deeper insights into LLM performance. Explore the paper for detailed information on its design, experiments, and implications. We invite you to leverage S3Eval for your research endeavors and contribute to the evolving landscape of synthetic benchmark construction. 😊

## Official link

### Paper

[S3Eval: A Synthetic, Scalable and Systematic Evaluation Suite for Large Language Models](arxiv.org/abs/2310.15147)

### Repository

[S3Eval](https://github.com/lfy79001/S3Eval)

## Examples
Input example I:
```

You are an SQL executor, you need to execute SQL based on the give table and SQL statement to obtain the execution results.
| suiting   | chisel    |   highboy |   broccoli | newburgh   | acetum    |   brewpub |
|:----------|:----------|----------:|-----------:|:-----------|:----------|----------:|
| zbwamhiui | nnkfvevxw |        50 |         88 | zhwohj     | opufj     |       214 |
| zroosgm   | yvftt     |       309 |        168 | zhwohj     | xqsu      |       136 |
| zroosgm   | lnri      |       152 |         78 | zhwohj     | ikvsd     |       219 |
| kjsdl     | trei      |       234 |        287 | egkgkvbec  | mhxcxyg   |        23 |
| zroosgm   | mctnpwbd  |        71 |        242 | egkgkvbec  | yszfokeom |       180 |
| zbwamhiui | ptqtj     |        19 |         81 | egkgkvbec  | hyfmk     |       116 |
| zroosgm   | lpjvwn    |       258 |        313 | uftnwbd    | oevmj     |        65 |
| kjsdl     | ididumrhw |        64 |        101 | uftnwbd    | xjakwpayx |       327 |
| zbwamhiui | wdtncbyn  |       165 |        209 | uftnwbd    | xrbqvxb   |       192 |
| zbwamhiui | wyjjc     |       219 |          6 | uftnwbd    | pzqr      |       188 |
| zroosgm   | qumxgwvls |       314 |        246 | uftnwbd    | ehevtf    |        60 |
| zbwamhiui | adiyf     |       207 |        298 | egkgkvbec  | wbrgejgf  |        80 |
| zbwamhiui | qpgpbj    |       307 |        306 | egkgkvbec  | mcjuonhc  |       192 |
| zbwamhiui | ehsk      |        47 |        244 | zhwohj     | tcdlnc    |       280 |
| kjsdl     | orlosbok  |        21 |         93 | egkgkvbec  | dzvwohjo  |       103 |
| zbwamhiui | webyyylw  |        84 |        195 | egkgkvbec  | xbmv      |       289 |
| kjsdl     | mrcecp    |        48 |        264 | egkgkvbec  | xhprcocik |       265 |
| kjsdl     | ngajupd   |       247 |         52 | zhwohj     | pcokyw    |       247 |
| zroosgm   | xeeuixkze |       120 |        288 | zhwohj     | yishnriw  |       138 |
| kjsdl     | kbczy     |       119 |         13 | egkgkvbec  | ltpmyfdt  |        73 |
| zbwamhiui | uvvdzo    |       150 |         57 | uftnwbd    | tajlsm    |       295 |
| zbwamhiui | enbffevhp |       290 |         92 | zhwohj     | gjjznp    |        18 |
| zroosgm   | imubtcc   |        79 |         19 | uftnwbd    | eqymwj    |       112 |

SQL:select suiting from my_table group by suiting having count ( newburgh ) > 6
Answer:
| suiting   |
|:----------|
| zbwamhiui |
| zroosgm   |

SQL:select acetum,newburgh,suiting from my_table where highboy > 234
Answer:
| acetum   | newburgh   | suiting   |
|:---------|:-----------|:----------|
| xqsu     | zhwohj     | zroosgm   |
| oevmj    | uftnwbd    | zroosgm   |
| ehevtf   | uftnwbd    | zroosgm   |
| mcjuonhc | egkgkvbec  | zbwamhiui |
| pcokyw   | zhwohj     | kjsdl     |
| gjjznp   | zhwohj     | zbwamhiui |

SQL:select count ( chisel ) from my_table where highboy < brewpub group by newburgh having min ( highboy ) < 47 
Answer:
|   count ( chisel ) |
|-------------------:|
|                  5 |

SQL:select newburgh from my_table where brewpub > 138 order by broccoli desc limit 1
Answer:
| newburgh   |
|:-----------|
| egkgkvbec  |


SQL:select suiting from my_table where highboy > broccoli group by suiting having min ( highboy ) < 314

Answer:

```
Output example I (from GPT-4):
```
| suiting   |
|:----------|
| kjsdl     |
| zbwamhiui |
| zroosgm   |

```

## Evaluation results

| LLM             | S3Eval-Easy | S3Eval-General | WikiTableQuestions  | BBH |
| --------------- | --------- | ------------ | ---- | -------------- |
| GPT-4           | **99.4**  | **63.1**     | 70.8 | 86.7           |
| ChatGPT         | 97.0      | 47.2         | 62.0 | 70.1           |
| Claude-1        | 98.2      | 44.3         | 63.4 | 67.3           |
| Llama-2-70B     | 94.2      | 41.3         | 55.9 | 64.9           |
| Mistral-7B      | 87.4      | 34.3         | 55.7 | 53.7           |
| Llama2-13B      | 75.0      | 30.9         | 49.2 | 45.6           |
| InternLM-20B    | 78.0      | 32.3         | 49.4 | 52.5           |
| Qwen-14B        | 71.8      | 25.8         | 46.7 | 53.7           |
| Llama-2-7B      | 54.2      | 23.8         | 40.6 | 38.2           |
| Qwen-7B         | 56.4      | 19.4         | 41.2 | 45.2           |
| Xgen-7B         | 55.2      | 24.6         | 36.3 | 34.5           |
| Internlm-7B     | 41.6      | 18.5         | 27.5 | 37.0           |
| Phi-1\_5        | 27.6      | 16.1         | 22.1 | 30.0           |
| Stablelm-7B     | 6.0       | 4.2          | 14.7 | 24.3           |
| Stablelm-3B     | 4.2       | 2.9          | 11.2 | 21.0           |
| Pythia-12B      | 31.4      | 17.3         | 24.5 | 29.3           |
| Pythia-6.9B     | 25.2      | 16.0         | 22.6 | 28.6           |
| Pythia-2.8B     | 26.4      | 14.6         | 21.7 | 28.8           |
| Pythia-1B       | 8.4       | 7.1          | 16.2 | 25.6           |

| Code LLM        | S3Eval-Easy | S3Eval-General | WikiTableQuestions | HumanEval |
| --------------- | --------- | ------------ | ---- | -------------- |
| CodeLlama-34B   | 91.4      | 41.0         | 53.9 | 36.4           |
| CodeLlama-13B   | 90.0      | 35.7         | 49.9 | 30.6           |
| CodeLlama-7B    | 75.2      | 34.2         | 44.9 | 26.3           |
| StarCoder-15B   | 87.2      | 34.4         | 39.2 | 30.4           |
| StarCoder-7B    | 88.4      | 32.4         | 33.3 | 28.3           |
| StarCoder-3B    | 79.0      | 28.0         | 27.5 | 21.5           |
| StarCoder-1B    | 37.4      | 15.4         | 21.1 | 15.2           |
| CodeGen-15B     | 36.8      | 18.2         | 25.0 | 18.3           |
| CodeGen-6B      | 25.0      | 16.9         | 17.8 | 18.2           |
| CodeGen-2B      | 31.4      | 16.6         | 20.8 | 14.5           |



## Reference
```
@article{lei2023s3eval,
  title={S3eval: A synthetic, scalable, systematic evaluation suite for large language models},
  author={Lei, Fangyu and Liu, Qian and Huang, Yiming and He, Shizhu and Zhao, Jun and Liu, Kang},
  journal={arXiv preprint arXiv:2310.15147},
  year={2023}
}
```