from mmengine.config import read_base

with read_base():
    from .datasets.dingo.dingo_gen import datasets
    from .models.hf_internlm.hf_internlm_7b import models

work_dir = './outputs/eval_dingo'
