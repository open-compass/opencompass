import os.path as osp

from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.musr.musr_gen_3c6e15 import musr_datasets
    from opencompass.configs.models.chatglm.lmdeploy_glm4_9b_chat import \
        models as lmdeploy_glm4_9b_chat_model
    from opencompass.configs.models.gemma.lmdeploy_gemma_9b_it import \
        models as lmdeploy_gemma_9b_it_model
    from opencompass.configs.models.gemma.lmdeploy_gemma_27b_it import \
        models as lmdeploy_gemma_27b_it_model
    # from opencompass.configs.models.hf_internlm.hf_internlm2_5_1_8b_chat import models
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import \
        models as lmdeploy_internlm2_5_7b_chat_model
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_1_8b_instruct import \
        models as lmdeploy_llama3_1_8b_instruct_model
    from opencompass.configs.models.mistral.lmdeploy_ministral_8b_instruct_2410 import \
        models as lmdeploy_ministral_8b_instruct_2410_model
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import \
        models as lmdeploy_qwen2_5_7b_instruct_model
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_14b_instruct import \
        models as lmdeploy_qwen2_5_14b_instruct_model
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_32b_instruct import \
        models as lmdeploy_qwen2_5_32b_instruct_model
    from opencompass.configs.models.yi.lmdeploy_yi_1_5_9b_chat import \
        models as lmdeploy_yi_1_5_9b_chat_model
    from opencompass.configs.summarizers.groups.musr_average import summarizer

datasets = [*musr_datasets]
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

base_exp_dir = 'outputs/musr/'
work_dir = osp.join(base_exp_dir, 'musr_eval')
