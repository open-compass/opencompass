from mmengine.config import read_base

from opencompass.models import OpenAISDK

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_gen import \
        race_datasets  # noqa: F401, E501
    # read hf models - chat models
    from opencompass.configs.models.baichuan.hf_baichuan2_7b_chat import \
        models as hf_baichuan2_7b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.chatglm.hf_glm4_9b_chat import \
        models as hf_glm4_9b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.hf_deepseek_7b_chat import \
        models as hf_deepseek_7b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.hf_deepseek_moe_16b_chat import \
        models as hf_deepseek_moe_16b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.vllm_deepseek_7b_chat import \
        models as vllm_deepseek_7b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.gemma.hf_gemma_2b_it import \
        models as hf_gemma_2b_it_model  # noqa: F401, E501
    from opencompass.configs.models.gemma.hf_gemma_7b_it import \
        models as hf_gemma_7b_it_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_5_7b_chat import \
        models as hf_internlm2_5_7b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import \
        models as lmdeploy_internlm2_5_7b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_chat_1_8b import \
        models as lmdeploy_internlm2_chat_1_8b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_chat_1_8b_sft import \
        models as lmdeploy_internlm2_chat_1_8b_sft_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_chat_7b import \
        models as lmdeploy_internlm2_chat_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_chat_7b_sft import \
        models as lmdeploy_internlm2_chat_7b_sft_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.vllm_internlm2_chat_7b import \
        models as vllm_internlm2_chat_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.hf_llama3_8b_instruct import \
        models as hf_llama3_8b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_8b_instruct import \
        models as lmdeploy_llama3_8b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.mistral.hf_mistral_7b_instruct_v0_2 import \
        models as hf_mistral_7b_instruct_v0_2_model  # noqa: F401, E501
    from opencompass.configs.models.mistral.vllm_mistral_7b_instruct_v0_2 import \
        models as vllm_mistral_7b_instruct_v0_2_model  # noqa: F401, E501
    from opencompass.configs.models.openbmb.hf_minicpm_2b_dpo_fp32 import \
        models as hf_minicpm_2b_dpo_fp32_model  # noqa: F401, E501
    from opencompass.configs.models.openbmb.hf_minicpm_2b_sft_bf16 import \
        models as hf_minicpm_2b_sft_bf16_model  # noqa: F401, E501
    from opencompass.configs.models.openbmb.hf_minicpm_2b_sft_fp32 import \
        models as hf_minicpm_2b_sft_fp32_model  # noqa: F401, E501
    from opencompass.configs.models.phi.hf_phi_3_mini_4k_instruct import \
        models as hf_phi_3_mini_4k_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.phi.hf_phi_3_small_8k_instruct import \
        models as hf_phi_3_mini_8k_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen1_5_0_5b_chat import \
        models as hf_qwen1_5_0_5b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen2_1_5b_instruct import \
        models as lmdeploy_qwen2_1_5b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen2_7b_instruct import \
        models as lmdeploy_qwen2_7b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.vllm_qwen1_5_0_5b_chat import \
        models as vllm_qwen1_5_0_5b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.yi.hf_yi_1_5_6b_chat import \
        models as hf_yi_1_5_6b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.yi.hf_yi_1_5_9b_chat import \
        models as hf_yi_1_5_9b_chat_model  # noqa: F401, E501
    from opencompass.configs.summarizers.medium import \
        summarizer  # noqa: F401, E501

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

model_name = ''

models.append(
    dict(
        abbr='lmdeploy-api-test',
        type=OpenAISDK,
        key='EMPTY',
        openai_api_base='http://judgemodel:10001/v1',
        path='compass_judger_internlm2_102b_0508',
        tokenizer_path='internlm/internlm2_5-20b-chat',
        rpm_verbose=True,
        meta_template=api_meta_template,
        query_per_second=50,
        max_out_len=1024,
        max_seq_len=4096,
        temperature=0.01,
        batch_size=128,
        retry=3,
    ))

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:100]'
