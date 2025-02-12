from mmengine.config import read_base

with read_base():
    # Models
    # Datasets
    from opencompass.configs.datasets.longbenchv2.longbenchv2_gen import \
        LongBenchv2_datasets as LongBenchv2_datasets
    from opencompass.configs.models.chatglm.lmdeploy_glm4_9b_chat import \
        models as lmdeploy_glm4_9b_chat_model
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_1_8b_instruct import \
        models as lmdeploy_llama3_1_8b_instruct_model
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import \
        models as lmdeploy_qwen2_5_7b_instruct_model

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

for model in models:
    model['max_seq_len'] = 128 * 1024
    model['engine_config']['session_len'] = 128 * 1024
    model['engine_config']['tp'] = 2
    model['run_cfg']['num_gpus'] = 2
    # Drop middle tokens to make input length shorter than session_len, use 128k to keep sync with Longbenchv2 original code
    # Drop middle now only support LMDeploy models
    model['drop_middle'] = True

work_dir = './outputs/longbenchv2'
