from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_nc_gen_c84c18 import nc_datasets
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_pp_acc_gen_8607a3 import pp_acc_datasets
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_rmse_gen_0fcc6b import pp_rmse_datasets
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_fts_gen_5774b5 import fts_datasets
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_meteor_gen_065150 import meteor_datasets

smolinstruct_datasets = nc_datasets + pp_rmse_datasets + pp_acc_datasets + meteor_datasets + fts_datasets
