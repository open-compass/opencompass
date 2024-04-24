# 数据污染评估

**数据污染** 是指本应用在下游测试任务重的数据出现在了大语言模型 (LLM) 的训练数据中，从而导致在下游任务 (例如，摘要、自然语言推理、文本分类) 上指标虚高，无法反映模型真实泛化能力的现象。

由于数据污染的源头是出现在 LLM 所用的训练数据中，因此最直接的检测数据污染的方法就是将测试数据与训练数据进行碰撞，然后汇报两者之间有多少语料是重叠出现的，经典的 GPT-3 [论文](https://arxiv.org/pdf/2005.14165.pdf)中的表 C.1 会报告了相关内容。

但如今开源社区往往只会公开模型参数而非训练数据集，在此种情况下 如何判断是否存在数据污染问题或污染程度如何，这些问题还没有被广泛接受的解决方案。OpenCompass 提供了两种可能的解决方案。

## 基于自建同分布数据的污染数据标注

我们参考了 [Skywork](https://arxiv.org/pdf/2310.19341.pdf) 中 5.2 节提到的方法，直接使用了 Skywork 上传到 HuggingFace 上的数据集 [mock_gsm8k_test](https://huggingface.co/datasets/Skywork/mock_gsm8k_test)。

在该方法中，作者使用 GPT-4 合成了一批与原始 GSM8K 风格类似的数据，然后使用模型分别计算在 GSM8K 训练集 (train)，GSM8K 测试集 (test)，GSM8K 参考集 (ref) 上的困惑度。由于 GSM8K 参考集是最新生成的，作者认为它必然不属于任何模型的任何训练集中，即它是干净的。作者认为：

- 若 测试集 的困惑度远小于 参考集 的困惑度，那么 测试集 可能出现在了模型的训练阶段；
- 若 训练集 的困惑度远小于 测试集 的困惑度，那么 训练集 可能被模型过拟合了。

我们可以参考使用以下配置文件:

```python
from mmengine.config import read_base

with read_base():
    from .datasets.gsm8k_contamination.gsm8k_contamination_ppl_ecdd22 import gsm8k_datasets  # 包含训练、测试、参考集
    from .models.qwen.hf_qwen_7b import models as hf_qwen_7b_model  # 待审查的模型
    from .models.yi.hf_yi_6b import models as hf_yi_6b_model

datasets = [*gsm8k_datasets]
models = [*hf_qwen_7b_model, *hf_yi_6b_model]
```

其样例输出如下：

```text
dataset          version    metric       mode       internlm-7b-hf    qwen-7b-hf    yi-6b-hf    chatglm3-6b-base-hf    qwen-14b-hf    baichuan2-13b-base-hf    internlm-20b-hf    aquila2-34b-hf  ...
---------------  ---------  -----------  -------  ----------------  ------------  ----------  ---------------------  -------------  -----------------------  -----------------  ----------------  ...
gsm8k-train-ppl  0b8e46     average_ppl  unknown              1.5           0.78        1.37                   1.16           0.5                      0.76               1.41              0.78  ...
gsm8k-test-ppl   0b8e46     average_ppl  unknown              1.56          1.33        1.42                   1.3            1.15                     1.13               1.52              1.16  ...
gsm8k-ref-ppl    f729ba     average_ppl  unknown              1.55          1.2         1.43                   1.35           1.27                     1.19               1.47              1.35  ...
```

目前该方案仅支持 GSM8K 数据集，我们欢迎社区贡献更多的数据集。

如果使用了该方法，请添加引用:

```bibtex
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}
@misc{wei2023skywork,
      title={Skywork: A More Open Bilingual Foundation Model},
      author={Tianwen Wei and Liang Zhao and Lichang Zhang and Bo Zhu and Lijie Wang and Haihua Yang and Biye Li and Cheng Cheng and Weiwei Lü and Rui Hu and Chenxia Li and Liu Yang and Xilin Luo and Xuejie Wu and Lunan Liu and Wenjun Cheng and Peng Cheng and Jianhao Zhang and Xiaoyu Zhang and Lei Lin and Xiaokun Wang and Yutuan Ma and Chuanhai Dong and Yanqi Sun and Yifu Chen and Yongyi Peng and Xiaojuan Liang and Shuicheng Yan and Han Fang and Yahui Zhou},
      year={2023},
      eprint={2310.19341},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## 基于经典预训练集的污染数据标注

感谢 [Contamination_Detector](https://github.com/liyucheng09/Contamination_Detector) 以及 @liyucheng09 提供了本方法。

在该方法中，作者将测试数据集 (例如 C-Eval, ARC, HellaSwag 等) 使用 Common Crawl 数据库和 Bing 搜索引擎来进行检索，然后依次标记每条测试样本是 干净的 / 题目被污染的 / 题目和答案均被污染的。

测试时，OpenCompass 会分别汇报 ceval 在三种标签所组成的子集上的准确率或困惑度。一般来说，准确率从低到高依次是 干净的，题目被污染的，题目和答案均被污染的 子集。作者认为：

- 若三者性能较为接近，则模型在该测试集上的污染程度较轻；反之则污染程度较重。

我们可以参考使用以下配置文件 [link](https://github.com/open-compass/opencompass/blob/main/configs/eval_contamination.py)：

```python
from mmengine.config import read_base

with read_base():
    from .datasets.ceval.ceval_clean_ppl import ceval_datasets  # 有污染标记的 ceval 数据集
    from .models.yi.hf_yi_6b import models as hf_yi_6b_model  # 待审查的模型
    from .models.qwen.hf_qwen_7b import models as hf_qwen_7b_model
    from .summarizers.contamination import ceval_summarizer as summarizer  # 输出格式整理

datasets = [*ceval_datasets]
models = [*hf_yi_6b_model, *hf_qwen_7b_model]
```

其样例输出如下：

```text
dataset                                         version    mode    yi-6b-hf          -                              -                                        qwen-7b-hf        -                              -                                        ...
----------------------------------------------  ---------  ------  ----------------  -----------------------------  ---------------------------------------  ----------------  -----------------------------  ---------------------------------------  ...
-                                               -          -       accuracy - clean  accuracy - input contaminated  accuracy - input-and-label contaminated  accuracy - clean  accuracy - input contaminated  accuracy - input-and-label contaminated  ...
...
ceval-humanities                                -          ppl     74.42             75.00                          82.14                                    67.44             50.00                          70.54                                    ...
ceval-stem                                      -          ppl     53.70             57.14                          85.61                                    47.41             52.38                          67.63                                    ...
ceval-social-science                            -          ppl     81.60             84.62                          83.09                                    76.00             61.54                          72.79                                    ...
ceval-other                                     -          ppl     72.31             73.91                          75.00                                    58.46             39.13                          61.88                                    ...
ceval-hard                                      -          ppl     44.35             37.50                          70.00                                    41.13             25.00                          30.00                                    ...
ceval                                           -          ppl     67.32             71.01                          81.17                                    58.97             49.28                          67.82                                    ...
```

目前该方案仅支持 C-Eval, MMLU, HellaSwag 和 ARC 数据集，[Contamination_Detector](https://github.com/liyucheng09/Contamination_Detector) 中还包含了 CSQA 和 WinoGrande，但目前还没有在 OpenCompass 中实现。我们欢迎社区贡献更多的数据集。

如果使用了该方法，请添加引用:

```bibtex
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}
@article{Li2023AnOS,
  title={An Open Source Data Contamination Report for Llama Series Models},
  author={Yucheng Li},
  journal={ArXiv},
  year={2023},
  volume={abs/2310.17589},
  url={https://api.semanticscholar.org/CorpusID:264490711}
}
```
