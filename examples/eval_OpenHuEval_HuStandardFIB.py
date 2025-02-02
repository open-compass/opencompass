from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.OpenHuEval.HuStandardFIB.HuStandardFIB import hu_standard_fib_datasets

    from opencompass.configs.models.openai.gpt_4o_mini_20240718 import models as gpt_4o_mini_20240718_model
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import models as lmdeploy_qwen2_5_7b_instruct_model
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm3_8b_instruct import models as lmdeploy_internlm3_8b_instruct_model
    # from opencompass.configs.models.qwq.lmdeploy_qwq_32b_preview import models as lmdeploy_qwq_32b_preview_model

datasets = hu_standard_fib_datasets
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
work_dir = './outputs/' + __file__.split('/')[-1].split('.')[0] + '/' # do NOT modify this line, yapf: disable, pylint: disable
