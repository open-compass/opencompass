# 大海捞针(Needle In A Haystack)实验评估

## 大海捞针测试简介

大海捞针测试（灵感来自[NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/LLMNeedleHaystackTester.py)）是一种评估方法，它通过在长文本中随机插入关键信息，形成大型语言模型(LLM)的Prompt。该测试旨在检测大型模型是否能从长文本中提取出这些关键信息，从而评估模型处理长文本信息提取的能力，这可以反映LLM对长文本的理解基础能力。

## 任务介绍

在`OpenCompass`的`NeedleBench`框架中，为了全面评估模型在长文本信息提取和推理方面的能力，我们设计了一系列逐渐增加难度的测试方案。完整的介绍参见我们的[技术报告](https://arxiv.org/abs/2407.11963)。

- **单一信息检索任务(Single-Needle Retrieval Task, S-RT)**：评估LLM在长文本中提取单一关键信息的能力，测试其对广泛叙述中特定细节的精确回忆能力。这对应于**原始的大海捞针测试**任务设定。

- **多信息检索任务(Multi-Needle Retrieval Task, M-RT)**：探讨LLM从长文本中检索多个相关信息的能力，模拟实际场景中对综合文档的复杂查询。

- **多信息推理任务(Multi-Needle Reasoning Task, M-RS)**：通过提取并利用长文本中的多个关键信息来评估LLM的长文本能力，要求模型对各关键信息片段有综合理解。

- **祖先追溯挑战(Ancestral Trace Challenge, ATC)**：通过设计“亲属关系针”，测试LLM处理真实长文本中多层逻辑挑战的能力。在ATC任务中，通过一系列逻辑推理问题，检验模型对长文本中每个细节的记忆和分析能力，在此任务中，我们去掉了无关文本(Haystack)的设定，而是将所有文本设计为关键信息，LLM必须综合运用长文本中的所有内容和推理才能准确回答问题。

### 评估步骤

> 注意：在最新代码中，OpenCompass已经设置数据集从[Huggingface的接口](https://huggingface.co/datasets/opencompass/NeedleBench)中自动加载，可以直接跳过下面的手动下载安放数据集。

1. 从[这里](https://github.com/open-compass/opencompass/files/14741330/needlebench.zip)下载数据集。

2. 将下载的文件放置于`opencompass/data/needlebench/`目录下。`needlebench`目录中预期的文件结构如下所示：

```
opencompass/
├── configs
├── docs
├── data
│   └── needlebench
│       ├── multi_needle_reasoning_en.json
│       ├── multi_needle_reasoning_zh.json
│       ├── names.json
│       ├── needles.jsonl
│       ├── PaulGrahamEssays.jsonl
│       ├── zh_finance.jsonl
│       ├── zh_game.jsonl
│       ├── zh_government.jsonl
│       ├── zh_movie.jsonl
│       ├── zh_tech.jsonl
│       ├── zh_general.jsonl
├── LICENSE
├── opencompass
├── outputs
├── run.py
├── more...
```

### `OpenCompass`环境配置

```bash
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```

### 配置数据集

我们在`configs/datasets/needlebench`中已经预先配置好了关于常见长度区间(4k, 8k, 32k, 128k, 200k, 1000k)的长文本测试设定，您可以通过在配置文件中定义相关参数，以灵活地创建适合您需求的数据集。

### 评估示例

#### 使用`LMDeploy`部署的 `InternLM2-7B` 模型进行评估

例如，使用`LMDeploy`部署的 `InternLM2-7B` 模型进行评估NeedleBench-4K的所有任务，可以在命令行中直接使用以下命令，该命令会调用预定义好的模型、数据集配置文件，而无需额外书写配置文件：

##### 本地评估

如果您在本地评估模型，下面命令会调用机器的所有可用GPU。您可以通过设置 `CUDA_VISIBLE_DEVICES` 环境变量来限制 `OpenCompass` 的 GPU 访问。例如，使用 `CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py ...` 只会向 OpenCompass 暴露前四个 GPU，确保它同时使用的 GPU 数量不超过这四个。

```bash
# 本地评估
python run.py --dataset needlebench_4k --models lmdeploy_internlm2_chat_7b  --summarizer needlebench/needlebench_4k_summarizer
```

##### 在Slurm集群上评估

如果使用 `Slurm`，可以添加 `--slurm -p partition_name -q reserved --max-num-workers 16 `等参数，例如下面：

```bash
# Slurm评估
python run.py --dataset needlebench_4k --models lmdeploy_internlm2_chat_7b  --summarizer needlebench/needlebench_4k_summarizer --slurm -p partition_name -q reserved --max-num-workers 16
```

##### 只评估子数据集

如果只想测试原始的大海捞针任务设定，比如可以更换数据集的参数为`needlebench_single_4k`，这对应于4k长度下的单针版本的大海捞针测试：

```bash
python run.py --dataset needlebench_single_4k --models lmdeploy_internlm2_chat_7b  --summarizer needlebench/needlebench_4k_summarizer --slurm -p partition_name -q reserved --max-num-workers 16
```

您也可以进一步选择子数据集，如更换数据集`--datasets`的参数为`needlebench_single_4k/needlebench_zh_datasets`，仅仅进行中文版本的单针4K长度下的大海捞针任务测试，其中`/`后面的参数代表子数据集，您可以在`configs/datasets/needlebench/needlebench_4k/needlebench_single_4k.py`中找到可选的子数据集变量，如：

```bash
python run.py --dataset needlebench_single_4k/needlebench_zh_datasets --models lmdeploy_internlm2_chat_7b  --summarizer needlebench/needlebench_4k_summarizer --slurm -p partition_name -q reserved --max-num-workers 16
```

注意在评估前预先安装[LMDeploy](https://github.com/InternLM/lmdeploy)工具

```bash
pip install lmdeploy
```

这个命令将启动评估流程，参数 `-p partition_name -q auto` 和 `--max-num-workers 32` 用于指定 Slurm 分区名称和最大工作进程数。

#### 评估其他`Huggingface`模型

对于其他模型，我们建议额外书写一个运行的配置文件以便对模型的`max_seq_len`, `max_out_len`参数进行修改，以便模型可以接收到完整的长文本内容。如我们预先写好的`configs/eval_needlebench.py`文件。完整内容如下

```python
from mmengine.config import read_base
# 我们使用mmengine.config来import其他的配置文件中的变量

with read_base():
    # from .models.hf_internlm.lmdeploy_internlm2_chat_7b import models as internlm2_chat_7b_200k
    from .models.hf_internlm.hf_internlm2_chat_7b import models as internlm2_chat_7b

    # Evaluate needlebench_4k, adjust the configuration to use 8k, 32k, 128k, 200k, or 1000k if necessary.
    # from .datasets.needlebench.needlebench_4k.needlebench_4k import needlebench_datasets
    # from .summarizers.needlebench import needlebench_4k_summarizer as summarizer

    # only eval original "needle in a haystack test" in needlebench_4k
    from .datasets.needlebench.needlebench_4k.needlebench_single_4k import needlebench_zh_datasets, needlebench_en_datasets
    from .summarizers.needlebench import needlebench_4k_summarizer as summarizer

    # eval Ancestral Tracing Challenge(ATC)
    # from .datasets.needlebench.atc.atc_choice_50 import needlebench_datasets
    # from .summarizers.needlebench import atc_summarizer_50 as summarizer

datasets = sum([v for k, v in locals().items() if ('datasets' in k)], [])

for m in internlm2_chat_7b:
    m['max_seq_len'] = 30768 # 保证InternLM2-7B模型能接收到完整的长文本，其他模型需要根据各自支持的最大序列长度修改。
    m['max_out_len'] = 2000 # 保证在多针召回任务中能接收到模型完整的回答

models = internlm2_chat_7b

work_dir = './outputs/needlebench'
```

当书写好测试的`config`文件后，我们可以命令行中通过`run.py`文件传入对应的config文件路径，例如：

```bash
python run.py configs/eval_needlebench.py  --slurm -p partition_name -q reserved --max-num-workers 16
```

注意，此时我们不需传入`--dataset, --models, --summarizer `等参数，因为我们已经在config文件中定义了这些配置。你可以自己手动调节`--max-num-workers`的设定以调节并行工作的workers的数量。

### 可视化

我们已经在最新的代码中将结果可视化内置到`summarizer`实现中，您在对应的output文件夹的plots目录下可以看到相应的可视化。而不需要自己手动可视化各个深度和长度下的分数。

如果使用了该方法，请添加引用:

```bibtex

@misc{li2024needlebenchllmsretrievalreasoning,
      title={NeedleBench: Can LLMs Do Retrieval and Reasoning in 1 Million Context Window?},
      author={Mo Li and Songyang Zhang and Yunxin Liu and Kai Chen},
      year={2024},
      eprint={2407.11963},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.11963},
}

@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished={\url{https://github.com/open-compass/opencompass}},
    year={2023}
}

@misc{LLMTest_NeedleInAHaystack,
  title={LLMTest Needle In A Haystack - Pressure Testing LLMs},
  author={gkamradt},
  year={2023},
  howpublished={\url{https://github.com/gkamradt/LLMTest_NeedleInAHaystack}}
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
