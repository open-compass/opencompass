# InstructBLIP

### Prepare the environment

```sh
git clone https://github.com/salesforce/LAVIS.git
cd ./LAVIS
pip install -e .
```

### Modify the config

Modify the config of InstructBlip, like model path of LLM and Qformer.

Then update `tasks.py` like the following code snippet.

```python
from mmengine.config import read_base

with read_base():
    from .instructblip.instructblip_mmbench import (instruct_blip_dataloader,
                                                    instruct_blip_evaluator,
                                                    instruct_blip_load_from,
                                                    instruct_blip_model)

models = [instruct_blip_model]
datasets = [instruct_blip_dataloader]
evaluators = [instruct_blip_evaluator]
load_froms = [instruct_blip_load_from]
num_gpus = 8
num_procs = 8
launcher = 'pytorch'  # or 'slurm'
```

### Start evaluation

#### Slurm

```sh
cd $root
python run.py configs/multimodal/tasks.py --mm-eval --slurm -p $PARTITION
```

#### PyTorch

```sh
cd $root
python run.py configs/multimodal/tasks.py --mm-eval 
```
