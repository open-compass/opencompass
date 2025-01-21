from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.OpenHuEval.HuMatchingFIB.HuMatchingFIB import FIB1_datasets

    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import models as lmdeploy_internlm2_5_7b_chat_model


datasets = FIB1_datasets
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
work_dir = './outputs/' + __file__.split('/')[-1].split('.')[0] + '/' # do NOT modify this line, yapf: disable, pylint: disable
