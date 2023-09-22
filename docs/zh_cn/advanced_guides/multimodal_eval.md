# 多模态评测

我们支持了多个多模态数据集，例如 [MMBench](https://opencompass.org.cn/MMBench)，[SEED-Bench](https://github.com/AILab-CVC/SEED-Bench)，来对多模态模型进行评测。在开始评测之前，请确保您已经按照官方教程下载了评测数据集。

## 开始评测

在评测前，您需要先修改 `tasks.py` 或者创建一个类似的新文件 `tasks_your_model.py` 来对您的模型进行评测。

一般来说我们使用下列命令启动评测。

### Slurm

```sh
cd $root
python run.py configs/multimodal/tasks.py --mm-eval --slurm -p $PARTITION
```

### PyTorch

```sh
cd $root
python run.py configs/multimodal/tasks.py --mm-eval
```

## 配置文件

We adapt the new config format of [MMEngine](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta).

### 任务文件

这是 `configs/multimodal/tasks.py` 的示例。

```python
from mmengine.config import read_base

with read_base():
    from .minigpt_4.minigpt_4_7b_mmbench import (minigpt_4_mmbench_dataloader,
                                                 minigpt_4_mmbench_evaluator,
                                                 minigpt_4_mmbench_load_from,
                                                 minigpt_4_mmbench_model)

models = [minigpt_4_mmbench_model]
datasets = [minigpt_4_mmbench_dataloader]
evaluators = [minigpt_4_mmbench_evaluator]
load_froms = [minigpt_4_mmbench_load_from]

# set the platform and resources
num_gpus = 8
num_procs = 8
launcher = 'pytorch'
```

### 细节配置

这是使用 MMBench 对 MiniGPT-4 进行评测的示例，我们提供了部分注释方便用户理解配置文件的含义。

```python
from opencompass.multimodal.models.minigpt_4 import (
    MiniGPT4MMBenchPromptConstructor, MiniGPT4MMBenchPostProcessor)

# dataloader settings
# 我们使用 MMPreTrain 中的 transforms 对图像数据进行处理
val_pipeline = [
    dict(type='mmpretrain.torchvision/Resize',
         size=(224, 224),
         interpolation=3),
    dict(type='mmpretrain.torchvision/ToTensor'),
    dict(type='mmpretrain.torchvision/Normalize',
         mean=(0.48145466, 0.4578275, 0.40821073),
         std=(0.26862954, 0.26130258, 0.27577711)),
    dict(type='mmpretrain.PackInputs',
         algorithm_keys=[
             'question', 'category', 'l2-category', 'context', 'index',
             'options_dict', 'options', 'split'
         ])
]

# 定义 MMBench dataset 来读取对应的数据
dataset = dict(type='opencompass.MMBenchDataset',
               data_file='data/mmbench/mmbench_test_20230712.tsv',
               pipeline=val_pipeline)

minigpt_4_mmbench_dataloader = dict(batch_size=1,
                                    num_workers=4,
                                    dataset=dataset,
                                    collate_fn=dict(type='pseudo_collate'),
                                    sampler=dict(type='DefaultSampler',
                                                 shuffle=False))

# model settings
minigpt_4_mmbench_model = dict(
    type='minigpt-4',  # 被测试的多模模型，type 在 `opencompass/multimodal/models/minigpt_4.py` 的 `@MM_MODELS.register_module('minigpt-4')` 中有定义
    low_resource=False,
    llama_model='/path/to/vicuna-7b/',  # LLM 的模型路径
    prompt_constructor=dict(type=MiniGPT4MMBenchPromptConstructor,  # 使用 PromptConstructor 来构建 LLM 的输入 prompt
                            image_prompt='###Human: <Img><ImageHere></Img>',
                            reply_prompt='###Assistant:'),
    post_processor=dict(type=MiniGPT4MMBenchPostProcessor))  # 使用 PostProcessor 来处理模型输出，使其符合输出格式的要求

# evaluation settings
minigpt_4_mmbench_evaluator = [
    dict(type='opencompass.DumpResults',  # evaluator 将结果保存在 save_path，代码在 `opencompass/metrics/dump_results.py`
         save_path='work_dirs/minigpt-4-7b-mmbench.xlsx')
]

minigpt_4_mmbench_load_from = '/path/to/prerained_minigpt4_7b.pth'  # 线性层的模型路径（MiniGPT-4 中 Q-Former 和 LLM 之间的线性投影层）
```
