# MMLUArabic
## Dataset Description
MMLUArabic is a benchmark for the assessment of knowledge in Arabic and covers a wide range of topics and aspects, consisting of multiple-choice questions in various branches of knowledge.


## How to Use
Download file from [link](https://github.com/FreedomIntelligence/AceGPT/tree/main/eval/benchmark_eval/benchmarks/MMLUArabic)

```python
val_ds = load_dataset("MMLUArabic", header=None)['validation']
test_ds = load_dataset("MMLUArabic", header=None)['test']
# input, option_a, option_b, option_c, option_d, target
print(next(iter(val_ds)))
```

## Citation
```
@misc{huang2023acegpt,
      title={AceGPT, Localizing Large Language Models in Arabic},
      author={Huang Huang and Fei Yu and Jianqing Zhu and Xuening Sun and Hao Cheng and Dingjie Song and Zhihong Chen and Abdulmohsen Alharthi and Bang An and Ziche Liu and Zhiyi Zhang and Junying Chen and Jianquan Li and Benyou Wang and Lian Zhang and Ruoyu Sun and Xiang Wan and Haizhou Li and Jinchao Xu},
      year={2023},
      eprint={2309.12053},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
