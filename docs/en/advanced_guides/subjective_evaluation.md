# Subjective Evaluation Guidance

## Introduction

Subjective evaluation aims to assess the model's performance in tasks that align with human preferences. The key criterion for this evaluation is human preference, but it comes with a high cost of annotation.

To explore the model's subjective capabilities, we employ JudgeLLM as a substitute for human assessors ([LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)).

A popular evaluation method involves

- Compare Mode: comparing model responses pairwise to calculate their win rate
- Score Mode: another method involves calculate scores with single model response ([Chatbot Arena](https://chat.lmsys.org/)).

We support the use of GPT-4 (or other JudgeLLM) for the subjective evaluation of models based on above methods.

## Subjective Evaluation with Custom Dataset

The specific process includes:

1. Data preparation
2. Model response generation
3. Evaluate the response with a JudgeLLM
4. Generate JudgeLLM's response and calculate the metric

### Step-1: Data Preparation

We provide mini test-set for **Compare Mode** and **Score Mode** as below:

```python
###COREV2
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

###CreationV0.1
[
    {
        "question": "请你扮演一个邮件管家，我让你给谁发送什么主题的邮件，你就帮我扩充好邮件正文，并打印在聊天框里。你需要根据我提供的邮件收件人以及邮件主题，来斟酌用词，并使用合适的敬语。现在请给导师发送邮件，询问他是否可以下周三下午15:00进行科研同步会，大约200字。",
        "capability": "邮件通知",
        "others": ""
    },
```

The json must includes the following fields:

- 'question': Question description
- 'capability': The capability dimension of the question.
- 'others': Other needed information.

If you want to modify prompt on each single question, you can full some other information into 'others' and construct it.

### Step-2: Evaluation Configuration(Compare Mode)

For `config/eval_subjective_compare.py`, we provide some annotations to help users understand the configuration file.

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
    # Pre-defined models
    from .models.qwen.hf_qwen_7b_chat import models as hf_qwen_7b_chat
    from .models.chatglm.hf_chatglm3_6b import models as hf_chatglm3_6b
    from .models.qwen.hf_qwen_14b_chat import models as hf_qwen_14b_chat
    from .models.openai.gpt_4 import models as gpt4_model
    from .datasets.subjective_cmp.subjective_corev2 import subjective_datasets

# Evaluation datasets
datasets = [*subjective_datasets]

# Model to be evaluated
models = [*hf_qwen_7b_chat, *hf_chatglm3_6b]

# Inference configuration
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llmeval',
        quotatype='auto',
        max_num_workers=256,
        task=dict(type=OpenICLInferTask)),
)
# Evaluation configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        mode='m2n', # m-model v.s n-model
        # Under m2n setting
        # must specify base_models and compare_models, program will generate pairs between base_models compare_models.
        base_models = [*hf_qwen_14b_chat], # Baseline model
        compare_models = [*hf_baichuan2_7b, *hf_chatglm3_6b] # model to be evaluated
    ),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llmeval',
        quotatype='auto',
        max_num_workers=256,
        task=dict(
            type=SubjectiveEvalTask,
        judge_cfg=gpt4_model # Judge model
        )),
)
work_dir = './outputs/subjective/'

summarizer = dict(
    type=Corev2Summarizer,  # Custom summarizer
    match_method='smart', # Answer extraction
)
```

In addition, you can also change the response order of the two models, please refer to `config/eval_subjective_compare.py`,
when `infer_order` is setting to `random`, the response will be random ordered,
when `infer_order` is setting to `double`, the response of two models will be doubled in two ways.

### Step-2: Evaluation Configuration(Score Mode)

For `config/eval_subjective_score.py`, it is mainly same with `config/eval_subjective_compare.py`, and you just need to modify the eval mode to `singlescore`.

### Step-3: Launch the Evaluation

```shell
python run.py config/eval_subjective_score.py -r
```

The `-r` parameter allows the reuse of model inference and GPT-4 evaluation results.

The response of JudgeLLM will be output to `output/.../results/timestamp/xxmodel/xxdataset/.json`.
The evaluation report will be output to `output/.../summary/timestamp/report.csv`.

Opencompass has supported lots of JudgeLLM, actually, you can take any model as JudgeLLM in opencompass configs.
And we list the popular open-source JudgeLLM here:

1. Auto-J, refer to `configs/models/judge_llm/auto_j`

Consider cite the following paper if you find it helpful:

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

2. JudgeLM, refer to `configs/models/judge_llm/judgelm`

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

3. PandaLM, refer to `configs/models/judge_llm/pandalm`

Consider cite the following paper if you find it helpful:

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

## Practice: AlignBench Evaluation

### Dataset

```bash
mkdir -p ./data/subjective/

cd ./data/subjective
git clone https://github.com/THUDM/AlignBench.git

# data format conversion
python ../../../tools/convert_alignmentbench.py --mode json --jsonl data/data_release.jsonl

```

### Configuration

Please edit the config `configs/eval_subjective_alignbench.py` according to your demand.

### Evaluation

```bash
HF_EVALUATE_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run.py workspace/eval_subjective_alignbench.py
```

### Submit to Official Leaderboard(Optional)

If you need to submit your prediction into official leaderboard, you can use `tools/convert_alignmentbench.py` for format conversion.

- Make sure you have the following results

```bash
outputs/
└── 20231214_173632
    ├── configs
    ├── logs
    ├── predictions # model's response
    ├── results
    └── summary
```

- Convert the data

```bash
python tools/convert_alignmentbench.py --mode csv --exp-folder outputs/20231214_173632
```

- Get `.csv`  in `submission/` for submission

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
