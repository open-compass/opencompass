from mmengine.config import read_base

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import \
        gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_ppl import \
        race_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.winogrande.winogrande_5shot_ll_252f01 import \
        winogrande_datasets  # noqa: F401, E501
    # read hf models - chat models
    from opencompass.configs.models.chatglm.lmdeploy_glm4_9b import \
        models as lmdeploy_glm4_9b_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.hf_deepseek_7b_base import \
        models as hf_deepseek_7b_base_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.lmdeploy_deepseek_7b_base import \
        models as lmdeploy_deepseek_7b_base_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.lmdeploy_deepseek_67b_base import \
        models as lmdeploy_deepseek_67b_base_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.lmdeploy_deepseek_v2 import \
        lmdeploy_deepseek_v2_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.vllm_deepseek_moe_16b_base import \
        models as vllm_deepseek_moe_16b_base_model  # noqa: F401, E501
    from opencompass.configs.models.gemma.hf_gemma2_2b import \
        models as hf_gemma2_2b_model  # noqa: F401, E501
    from opencompass.configs.models.gemma.hf_gemma2_9b import \
        models as hf_gemma2_9b_model  # noqa: F401, E501
    from opencompass.configs.models.gemma.hf_gemma_2b import \
        models as hf_gemma_2b_model  # noqa: F401, E501
    from opencompass.configs.models.gemma.hf_gemma_7b import \
        models as hf_gemma_7b_model  # noqa: F401, E501
    from opencompass.configs.models.gemma.lmdeploy_gemma_9b import \
        models as lmdeploy_gemma_9b_model  # noqa: F401, E501
    from opencompass.configs.models.gemma.vllm_gemma_2b import \
        models as vllm_gemma_2b_model  # noqa: F401, E501
    from opencompass.configs.models.gemma.vllm_gemma_7b import \
        models as vllm_gemma_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_5_7b import \
        models as hf_internlm2_5_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_7b import \
        models as hf_internlm2_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_1_8b import \
        models as lmdeploy_internlm2_1_8b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b import \
        models as lmdeploy_internlm2_5_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_7b import \
        models as lmdeploy_internlm2_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_20b import \
        models as lmdeploy_internlm2_20b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_base_7b import \
        models as lmdeploy_internlm2_base_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_base_20b import \
        models as lmdeploy_internlm2_base_20b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.hf_llama2_7b import \
        models as hf_llama2_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.hf_llama3_1_8b import \
        models as hf_llama3_1_8b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.hf_llama3_8b import \
        models as hf_llama3_8b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_1_8b import \
        models as lmdeploy_llama3_1_8b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_8b import \
        models as lmdeploy_llama3_8b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_70b import \
        models as lmdeploy_llama3_70b_model  # noqa: F401, E501
    from opencompass.configs.models.mistral.hf_mistral_7b_v0_3 import \
        models as hf_mistral_7b_v0_3_model  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.hf_qwen_2_5_7b import \
        models as hf_qwen_2_5_7b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.hf_qwen_2_5_14b import \
        models as hf_qwen_2_5_14b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.hf_qwen_2_5_32b import \
        models as hf_qwen_2_5_32b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_1_5b import \
        models as lmdeploy_qwen2_5_1_5b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b import \
        models as lmdeploy_qwen2_5_7b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_32b import \
        models as lmdeploy_qwen2_5_32b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_72b import \
        models as lmdeploy_qwen2_5_72b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen1_5_moe_a2_7b import \
        models as hf_qwen1_5_moe_a2_7b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen2_0_5b import \
        models as hf_qwen2_0_5b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen2_1_5b import \
        models as hf_qwen2_1_5b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen2_7b import \
        models as hf_qwen2_7b_model  # noqa: F401, E501
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
    from opencompass.configs.models.yi.lmdeploy_yi_1_5_9b import \
        models as lmdeploy_yi_1_5_9b_model  # noqa: F401, E501

    from ...volc import infer as volc_infer  # noqa: F401, E501

race_datasets = [race_datasets[1]]
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:32]'

for m in models:
    if 'turbomind' in m['abbr'] or 'lmdeploy' in m['abbr']:
        m['engine_config']['max_batch_size'] = 1
        m['batch_size'] = 1
models = sorted(models, key=lambda x: x['run_cfg']['num_gpus'])

summarizer = dict(
    dataset_abbrs=[
        ['gsm8k', 'accuracy'],
        ['GPQA_diamond', 'accuracy'],
        ['race-high', 'accuracy'],
        ['winogrande', 'accuracy'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
