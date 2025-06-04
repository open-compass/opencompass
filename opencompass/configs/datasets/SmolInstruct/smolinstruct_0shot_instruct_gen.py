from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_nc_0shot_instruct import nc_0shot_instruct_datasets
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_pp_acc_0_shot_instruct import pp_acc_datasets_0shot_instruct
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_rmse_0shot_instruct import pp_rmse_0shot_instruct_datasets
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_fts_0shot_instruct import fts_0shot_instruct_datasets
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_meteor_0shot_instruct import meteor_0shot_instruct_datasets

smolinstruct_datasets_0shot_instruct = nc_0shot_instruct_datasets + pp_rmse_0shot_instruct_datasets + pp_acc_datasets_0shot_instruct + meteor_0shot_instruct_datasets + fts_0shot_instruct_datasets
