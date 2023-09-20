from mmengine.config import read_base

with read_base():
    from .llava.llava_7b_vqav2 import (
        llava_vqav2_dataloader,
        llava_vqav2_evaluator,
        llava_vqav2_model,
    )

models = [llava_vqav2_model]
datasets = [llava_vqav2_dataloader]
evaluators = [llava_vqav2_evaluator]
load_froms = [None]

num_gpus = 8
num_procs = 8
launcher = 'pytorch'
