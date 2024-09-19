from mmengine.config import read_base

with read_base():
    from .models.hf_internlm.hf_internlm_7b import models
    from .datasets.dingo.dingo_gen import datasets

work_dir = './outputs/eval_dingo'
