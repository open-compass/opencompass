# S3Eval
## Introduction
The following benchmark comes from the paper in [S3eval: A synthetic, scalable, systematic evaluation suite for large language models](https://arxiv.org/abs/2310.15147)


S3Eval, our latest contribution to the field, addresses the critical need for comprehensive evaluation resources for Large Language Models (LLMs). In the pursuit of understanding long-context comprehension and enhancing reasoning capabilities, we present a benchmarking suite that is both synthetic and scalable.

Operating on SQL execution tasks, S3Eval challenges LLMs with randomly generated tables and SQL queries, evaluating their ability to produce accurate execution results. This benchmark stands out for its versatility and scalability, providing unlimited evaluation resources for a robust assessment of LLM capabilities.

In this latest submission, we have generated a batch of high-quality data, encompassing nearly all types of queries with strong diversity. Moreover, the length of the tables spans from 200 to 200K, enabling a systematic evaluation of the long-context capabilities of the models.

For researchers and practitioners alike, S3Eval holds the promise of uncovering deeper insights into LLM performance. Explore the paper for detailed information on its design, experiments, and implications. We invite you to leverage S3Eval for your research endeavors and contribute to the evolving landscape of synthetic benchmark construction. 😊


## Official link

### Paper

[S3eval: A synthetic, scalable, systematic evaluation suite for large language models](https://arxiv.org/abs/2310.15147)

### Repository

[s3eval](https://github.com/lfy79001/S3Eval)

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


| Model         | Score |
|---------------|-------|
| GPT-4         | 61.3  |
| GPT3.5-Turbo  | 40.2  |
| Code LLama 34B| 28.3  |
| Code LLama 13B| 21.5  |
| Code LLama 7B | 12.7  |
| Starcoder1 15B| 12.5  |
| Starcoder1 7B | 10.2  |
| Starcoder1 3B | 7.8   |
| Starcoder1 1B | 5.4   |
| Llama 13B     | 13.1  |
| Llama 7B      | 6.5   |
| Deepseek 7B   | 12.6  |
| Olmo 7B       | 8.2   |
| Qwen 14B      | 12.3  |
| Qwen 7B       | 11.6  |
| Mistral 7B    | 12.4  |
| Internlm 20B  | 14.6  |




## Reference
```
@article{lei2023s3eval,
  title={S3eval: A synthetic, scalable, systematic evaluation suite for large language models},
  author={Lei, Fangyu and Liu, Qian and Huang, Yiming and He, Shizhu and Zhao, Jun and Liu, Kang},
  journal={arXiv preprint arXiv:2310.15147},
  year={2023}
}
```
