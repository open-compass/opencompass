# Data Contamination Assessment

**Data Contamination** refers to the phenomenon where data intended for downstream testing tasks appear in the training data of large language models (LLMs), resulting in artificially inflated performance metrics in downstream tasks (such as summarization, natural language inference, text classification), which do not accurately reflect the model's true generalization capabilities.

Since the source of data contamination lies in the training data used by LLMs, the most direct method to detect data contamination is to collide test data with training data and then report the extent of overlap between the two. The classic GPT-3 [paper](https://arxiv.org/pdf/2005.14165.pdf) reported on this in Table C.1.

However, today's open-source community often only publishes model parameters, not training datasets. In such cases, how to determine the presence and extent of data contamination remains unsolved. OpenCompass offers two possible solutions.

## Contamination Data Annotation Based on Self-Built Co-Distribution Data

Referencing the method mentioned in Section 5.2 of [Skywork](https://arxiv.org/pdf/2310.19341.pdf), we directly used the dataset [mock_gsm8k_test](https://huggingface.co/datasets/Skywork/mock_gsm8k_test) uploaded to HuggingFace by Skywork.

In this method, the authors used GPT-4 to synthesize data similar to the original GSM8K style, and then calculated the perplexity on the GSM8K training set (train), GSM8K test set (test), and GSM8K reference set (ref). Since the GSM8K reference set was newly generated, the authors considered it as clean, not belonging to any training set of any model. They posited:

- If the test set's perplexity is significantly lower than the reference set's, the test set might have appeared in the model's training phase;
- If the training set's perplexity is significantly lower than the test set's, the training set might have been overfitted by the model.

The following configuration file can be referenced:

```python
from mmengine.config import read_base

with read_base():
    from .datasets.gsm8k_contamination.gsm8k_contamination_ppl_ecdd22 import gsm8k_datasets  # includes training, test, and reference sets
    from .models.qwen.hf_qwen_7b import models as hf_qwen_7b_model  # model under review
    from .models.yi.hf_yi_6b import models as hf_yi_6b_model

datasets = [*gsm8k_datasets]
models = [*hf_qwen_7b_model, *hf_yi_6b_model]
```

An example output is as follows:

```text
dataset          version    metric       mode       internlm-7b-hf    qwen-7b-hf    yi-6b-hf    chatglm3-6b-base-hf    qwen-14b-hf    baichuan2-13b-base-hf    internlm-20b-hf    aquila2-34b-hf  ...
---------------  ---------  -----------  -------  ----------------  ------------  ----------  ---------------------  -------------  -----------------------  -----------------  ----------------  ...
gsm8k-train-ppl  0b8e46     average_ppl  unknown              1.5           0.78        1.37                   1.16           0.5                      0.76               1.41              0.78  ...
gsm8k-test-ppl   0b8e46     average_ppl  unknown              1.56          1.33        1.42                   1.3            1.15                     1.13               1.52              1.16  ...
gsm8k-ref-ppl    f729ba     average_ppl  unknown              1.55          1.2         1.43                   1.35           1.27                     1.19               1.47              1.35  ...
```

Currently, this solution only supports the GSM8K dataset. We welcome the community to contribute more datasets.

Consider cite the following paper if you find it helpful:

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

## Contamination Data Annotation Based on Classic Pre-trained Sets

Thanks to [Contamination_Detector](https://github.com/liyucheng09/Contamination_Detector) and @liyucheng09 for providing this method.

In this method, the authors search the test datasets (such as C-Eval, ARC, HellaSwag, etc.) using the Common Crawl database and Bing search engine, then mark each test sample as clean / question contaminated / both question and answer contaminated.

During testing, OpenCompass

will report the accuracy or perplexity of ceval on subsets composed of these three labels. Generally, the accuracy ranges from low to high: clean, question contaminated, both question and answer contaminated subsets. The authors believe:

- If the performance of the three is relatively close, the contamination level of the model on that test set is light; otherwise, it is heavy.

The following configuration file can be referenced [link](https://github.com/open-compass/opencompass/blob/main/configs/eval_contamination.py):

```python
from mmengine.config import read_base

with read_base():
    from .datasets.ceval.ceval_clean_ppl import ceval_datasets  # ceval dataset with contamination tags
    from .models.yi.hf_yi_6b import models as hf_yi_6b_model  # model under review
    from .models.qwen.hf_qwen_7b import models as hf_qwen_7b_model
    from .summarizers.contamination import ceval_summarizer as summarizer  # output formatting

datasets = [*ceval_datasets]
models = [*hf_yi_6b_model, *hf_qwen_7b_model]
```

An example output is as follows:

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

Currently, this solution only supports the C-Eval, MMLU, HellaSwag and ARC. [Contamination_Detector](https://github.com/liyucheng09/Contamination_Detector) also includes CSQA and WinoGrande, but these have not yet been implemented in OpenCompass. We welcome the community to contribute more datasets.

Consider cite the following paper if you find it helpful:

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
