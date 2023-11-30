# Multi-modality Evaluation

We support several multi-modality datasets, such as [MMBench](https://opencompass.org.cn/MMBench), [SEED-Bench](https://github.com/AILab-CVC/SEED-Bench) to evaluate multi-modality models. Before starting, please make sure you have downloaded the evaluation datasets following the official instruction.

## Start Evaluation

Before evaluation, you could modify `tasks.py` or create a new file like `tasks.py` to evaluate your own model.

Generally to run the evaluation, we use command below.

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

## Configuration File

We adapt the new config format of [MMEngine](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta).

### Task File

Here is the example config of `configs/multimodal/tasks.py`.

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

### Details of Task

Here is an example of MiniGPT-4 with MMBench and we provide some comments for
users to understand the meaning of the keys in config.

```python
from opencompass.multimodal.models.minigpt_4 import (
    MiniGPT4MMBenchPromptConstructor, MiniGPT4MMBenchPostProcessor)

# dataloader settings
# Here we use Transforms in MMPreTrain to process images
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

# The defined MMBench datasets to load evaluation data
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
    type='minigpt-4',  # the test multomodal algorithm, the type can be found in `opencompass/multimodal/models/minigpt_4.py`, `@MM_MODELS.register_module('minigpt-4')`
    low_resource=False,
    llama_model='/path/to/vicuna-7b/',  # the model path of LLM
    prompt_constructor=dict(type=MiniGPT4MMBenchPromptConstructor,  # the PromptConstructor to construct the prompt
                            image_prompt='###Human: <Img><ImageHere></Img>',
                            reply_prompt='###Assistant:'),
    post_processor=dict(type=MiniGPT4MMBenchPostProcessor))  # the PostProcessor to deal with the output, process it into the required format

# evaluation settings
minigpt_4_mmbench_evaluator = [
    dict(type='opencompass.DumpResults',  # the evaluator will dump results to save_path, code can be found in `opencompass/metrics/dump_results.py`
         save_path='work_dirs/minigpt-4-7b-mmbench.xlsx')
]

minigpt_4_mmbench_load_from = '/path/to/prerained_minigpt4_7b.pth'  # the model path of linear layer between Q-Former and LLM in MiniGPT-4
```
