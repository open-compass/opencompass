# 主观评测指引

## 介绍

主观评测旨在评估模型在符合人类偏好的能力上的表现。这种评估的黄金准则是人类喜好，但标注成本很高。

为了探究模型的主观能力，我们采用了JudgeLLM作为人类评估者的替代品（[LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)）。

流行的评估方法主要有:

- Compare模式：将模型的回答进行两两比较，以计算对战其胜率。
- Score模式：针对单模型的回答进行打分（例如：[Chatbot Arena](https://chat.lmsys.org/)）。

我们基于以上方法支持了JudgeLLM用于模型的主观能力评估（目前opencompass仓库里支持的所有模型都可以直接作为JudgeLLM进行调用，此外一些专用的JudgeLLM我们也在计划支持中）。

## 自定义主观数据集评测

主观评测的具体流程包括:

1. 评测数据集准备
2. 使用API模型或者开源模型进行问题答案的推理
3. 使用选定的评价模型(JudgeLLM)对模型输出进行评估
4. 对评价模型返回的预测结果进行解析并计算数值指标

### 第一步：数据准备

对于对战模式和打分模式，我们各提供了一个demo测试集如下：

```python
### 对战模式示例
[
    {
        "question": "如果我在空中垂直抛球，球最初向哪个方向行进？",
        "capability": "知识-社会常识",
        "others": {
            "question": "如果我在空中垂直抛球，球最初向哪个方向行进？",
            "evaluating_guidance": "",
            "reference_answer": "上"
        }
    },...]

### 打分模式数据集示例
[
    {
        "question": "请你扮演一个邮件管家，我让你给谁发送什么主题的邮件，你就帮我扩充好邮件正文，并打印在聊天框里。你需要根据我提供的邮件收件人以及邮件主题，来斟酌用词，并使用合适的敬语。现在请给导师发送邮件，询问他是否可以下周三下午15:00进行科研同步会，大约200字。",
        "capability": "邮件通知",
        "others": ""
    },
```

如果要准备自己的数据集，请按照以下字段进行提供，并整理为一个json文件：

- 'question'：问题描述
- 'capability'：题目所属的能力维度
- 'others'：其他可能需要对题目进行特殊处理的项目

以上三个字段是必要的，用户也可以添加其他字段，如果需要对每个问题的prompt进行单独处理，可以在'others'字段中进行一些额外设置，并在Dataset类中添加相应的字段。

### 第二步：构建评测配置（对战模式）

对于两回答比较，更详细的config setting请参考 `config/eval_subjective_compare.py`，下面我们提供了部分简略版的注释，方便用户理解配置文件的含义。

```python
from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM, HuggingFace, OpenAI

from opencompass.partitioners import NaivePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import Corev2Summarizer

with read_base():
    # 导入预设模型
    from .models.qwen.hf_qwen_7b_chat import models as hf_qwen_7b_chat
    from .models.chatglm.hf_chatglm3_6b import models as hf_chatglm3_6b
    from .models.qwen.hf_qwen_14b_chat import models as hf_qwen_14b_chat
    from .models.openai.gpt_4 import models as gpt4_model
    from .datasets.subjective_cmp.subjective_corev2 import subjective_datasets

# 评测数据集
datasets = [*subjective_datasets]

# 待测模型列表
models = [*hf_qwen_7b_chat, *hf_chatglm3_6b]

# 推理配置
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llmeval',
        quotatype='auto',
        max_num_workers=256,
        task=dict(type=OpenICLInferTask)),
)
# 评测配置
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        mode='m2n', # m个模型 与 n个模型进行对战
        #  在m2n模式下，需要指定base_models和compare_models，将会对base_models和compare_models生成对应的两两pair（去重且不会与自身进行比较）
        base_models = [*hf_qwen_14b_chat], # 用于对比的基线模型
        compare_models = [*hf_baichuan2_7b, *hf_chatglm3_6b] # 待评测模型
    ),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llmeval',
        quotatype='auto',
        max_num_workers=256,
        task=dict(
            type=SubjectiveEvalTask,
        judge_cfg=gpt4_model # 评价模型
        )),
)
work_dir = './outputs/subjective/' #指定工作目录，在此工作目录下，若使用--reuse参数启动评测，将自动复用该目录下已有的所有结果

summarizer = dict(
    type=Corev2Summarizer,  #自定义数据集Summarizer
    match_method='smart', #自定义答案提取方式
)
```

此外，在数据集的配置config中，还可以选择两回答比较时的回答顺序，请参考`config/eval_subjective_compare.py`,
当`infer_order`设置为`random`时，将对两模型的回复顺序进行随机打乱,
当`infer_order`设置为`double`时，将把两模型的回复按两种先后顺序进行判断。

### 第二步：构建评测配置（打分模式）

对于单回答打分，更详细的config setting请参考 `config/eval_subjective_score.py`，该config的大部分都与两回答比较的config相同，只需要修改评测模式即可，将评测模式设置为`singlescore`。

### 第三步 启动评测并输出评测结果

```shell
python run.py configs/eval_subjective_score.py -r
```

- `-r` 参数支持复用模型推理和评估结果。

JudgeLLM的评测回复会保存在 `output/.../results/timestamp/xxmodel/xxdataset/.json`
评测报告则会输出到 `output/.../summary/timestamp/report.csv`。

Opencompass 已经支持了很多的JudgeLLM，实际上，你可以将Opencompass中所支持的所有模型都当作JudgeLLM使用。
我们列出目前比较流行的开源JudgeLLM：

1. Auto-J，请参考 `configs/models/judge_llm/auto_j`

如果使用了该方法，请添加引用:

```bibtex
@article{li2023generative,
  title={Generative judge for evaluating alignment},
  author={Li, Junlong and Sun, Shichao and Yuan, Weizhe and Fan, Run-Ze and Zhao, Hai and Liu, Pengfei},
  journal={arXiv preprint arXiv:2310.05470},
  year={2023}
}
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}
```

2. JudgeLM，请参考 `configs/models/judge_llm/judgelm`

如果使用了该方法，请添加引用:

```bibtex
@article{zhu2023judgelm,
  title={JudgeLM: Fine-tuned Large Language Models are Scalable Judges},
  author={Zhu, Lianghui and Wang, Xinggang and Wang, Xinlong},
  journal={arXiv preprint arXiv:2310.17631},
  year={2023}
}
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}
```

3. PandaLM，请参考 `configs/models/judge_llm/pandalm`

如果使用了该方法，请添加引用:

```bibtex
@article{wang2023pandalm,
  title={PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization},
  author={Wang, Yidong and Yu, Zhuohao and Zeng, Zhengran and Yang, Linyi and Wang, Cunxiang and Chen, Hao and Jiang, Chaoya and Xie, Rui and Wang, Jindong and Xie, Xing and others},
  journal={arXiv preprint arXiv:2306.05087},
  year={2023}
}
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}
```

## 实战：AlignBench 主观评测

### 数据集准备

```bash
mkdir -p ./data/subjective/

cd ./data/subjective
git clone https://github.com/THUDM/AlignBench.git

# data format conversion
python ../../../tools/convert_alignmentbench.py --mode json --jsonl data/data_release.jsonl

```

### 配置文件

请根据需要修改配置文件 `configs/eval_subjective_alignbench.py`

### 启动评测

按如下方式执行命令后，将会开始答案推理和主观打分，如只需进行推理，可以通过制定 `-m infer`实现

```bash
HF_EVALUATE_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run.py configs/eval_subjective_alignbench.py
```

### 提交官方评测（Optional）

完成评测后，如需提交官方榜单进行评测，可以使用它`tools/convert_alignmentbench.py`进行格式转换。

- 请确保已完成推理，并获得如下所示的文件:

```bash
outputs/
└── 20231214_173632
    ├── configs
    ├── logs
    ├── predictions # 模型回复
    ├── results
    └── summary
```

- 执行如下命令获得可用于提交的结果

```bash
python tools/convert_alignmentbench.py --mode csv --exp-folder outputs/20231214_173632
```

- 进入 `submission`文件夹获得可用于提交的`.csv`文件

```bash
outputs/
└── 20231214_173632
    ├── configs
    ├── logs
    ├── predictions
    ├── results
    ├── submission # 可提交文件
    └── summary
```
