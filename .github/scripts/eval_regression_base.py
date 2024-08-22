from mmengine.config import read_base

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_ppl import \
        race_datasets  # noqa: F401, E501
    from opencompass.configs.models.deepseek.hf_deepseek_moe_16b_base import \
        models as hf_deepseek_moe_16b_base_model  # noqa: F401, E501
    # read hf models - chat models
    from opencompass.configs.models.deepseek.lmdeploy_deepseek_7b_base import \
        models as lmdeploy_deepseek_7b_base_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.vllm_deepseek_moe_16b_base import \
        models as vllm_deepseek_moe_16b_base_model  # noqa: F401, E501
    from opencompass.configs.models.gemma.hf_gemma_2b import \
        models as hf_gemma_2b_model  # noqa: F401, E501
    from opencompass.configs.models.gemma.hf_gemma_7b import \
        models as hf_gemma_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_5_7b import \
        models as hf_internlm2_5_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_7b import \
        models as hf_internlm2_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_base_7b import \
        models as hf_internlm2_base_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_1_8b import \
        models as lmdeploy_internlm2_1_8b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b import \
        models as lmdeploy_internlm2_5_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_7b import \
        models as lmdeploy_internlm2_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_base_7b import \
        models as lmdeploy_internlm2_base_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_8b import \
        models as lmdeploy_llama3_8b_model  # noqa: F401, E501
    from opencompass.configs.models.mistral.hf_mistral_7b_v0_2 import \
        models as hf_mistral_7b_v0_2_model  # noqa: F401, E501
    from opencompass.configs.models.mistral.vllm_mistral_7b_v0_2 import \
        models as vllm_mistral_7b_v0_2_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen1_5_moe_a2_7b import \
        models as hf_qwen1_5_moe_a2_7b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen2_0_5b import \
        models as hf_qwen2_0_5b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen2_1_5b import \
        models as lmdeploy_qwen2_1_5b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen2_7b import \
        models as lmdeploy_qwen2_7b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.vllm_qwen1_5_0_5b import \
        models as vllm_qwen1_5_0_5b_model  # noqa: F401, E501
    from opencompass.configs.models.yi.hf_yi_1_5_6b import \
        models as hf_yi_1_5_6b_model  # noqa: F401, E501
    from opencompass.configs.models.yi.hf_yi_1_5_9b import \
        models as hf_yi_1_5_9b_model  # noqa: F401, E501
    from opencompass.configs.summarizers.medium import \
        summarizer  # noqa: F401, E501

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:100]'
