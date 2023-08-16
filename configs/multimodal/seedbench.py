from mmengine.config import read_base

with read_base():
    from .minigpt_4.minigpt_4_7b_seedbench import (
        minigpt_4_seedbench_dataloader, minigpt_4_seedbench_evaluator,
        minigpt_4_load_from, minigpt_4_seedbench_model)

models = [minigpt_4_seedbench_model]
datasets = [minigpt_4_seedbench_dataloader]
evaluators = [minigpt_4_seedbench_evaluator]
load_froms = [minigpt_4_load_from]
num_gpus = 1
num_procs = 1
launcher = 'slurm'
