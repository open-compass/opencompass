from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_nc_gen import nc_datasets
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_pp_acc_gen import pp_acc_datasets
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_rmse_gen import pp_rmse_datasets
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_fts_gen import fts_datasets
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_meteor_gen import meteor_datasets

smolinstruct_datasets = nc_datasets + pp_rmse_datasets + pp_acc_datasets + meteor_datasets + fts_datasets
