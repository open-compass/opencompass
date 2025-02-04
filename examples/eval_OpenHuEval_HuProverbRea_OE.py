from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.OpenHuEval.HuProverbRea.HuProverbRea_OE import HuProverbRea_datasets
    from opencompass.configs.models.qwen.lmdeploy_qwen2_7b_instruct import models as lmdeploy_qwen2_7b_instruct_model
    # from opencompass.configs.models.openai.gpt_4o_mini_20240718 import models as gpt_4o_mini_20240718_model
    # from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import models as lmdeploy_qwen2_5_7b_instruct_model
    # from opencompass.configs.models.hf_internlm.lmdeploy_internlm3_8b_instruct import models as lmdeploy_internlm3_8b_instruct_model

    # from opencompass.configs.models.qwq.lmdeploy_qwq_32b_preview import models as lmdeploy_qwq_32b_preview_model
    # from opencompass.configs.models.openai.o1_mini_2024_09_12 import models as o1_mini_2024_09_12_model
    # from opencompass.configs.models.openai.o3_mini_2025_01_31 import models as o3_mini_2025_01_31_model

datasets = HuProverbRea_datasets
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
work_dir = './outputs/' + __file__.split('/')[-1].split('.')[0] + '/' # do NOT modify this line, yapf: disable, pylint: disable
