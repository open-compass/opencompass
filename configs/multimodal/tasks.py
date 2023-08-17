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
num_gpus = 8
num_procs = 8
launcher = 'pytorch'