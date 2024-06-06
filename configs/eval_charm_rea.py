from mmengine.config import read_base

with read_base():
    from .datasets.CHARM.charm_reason_gen_f8fca2 import charm_reason_datasets as datasets

    # ------>>>>>> https://arxiv.org/abs/2403.14112
    # from .models.openai.gpt_3_5_turbo_1106 import models as gpt_3_5_turbo_1106_model
    # from .models.openai.gpt_4_1106_preview import models as gpt_4_1106_preview_model
    # from .models.hf_llama.hf_llama2_7b_chat import models as llama2_7b_chat_model
    # from .models.hf_llama.hf_llama2_13b_chat import models as llama2_13b_chat_model
    # from .models.hf_llama.hf_llama2_70b_chat import models as llama2_70b_chat_model
    # from .models.vicuna.hf_vicuna_7b_v15_16k import models as vicuna_7b_v15_16k_model
    # from .models.vicuna.hf_vicuna_13b_v15_16k import models as vicuna_13b_v15_16k_model
    # from .models.chatglm.hf_chatglm3_6b_32k import models as chatglm3_6b_32k_model
    # from .models.baichuan.hf_baichuan2_7b_chat import models as baichuan2_7b_chat_model  # need torch 2.1
    # from .models.baichuan.hf_baichuan2_13b_chat import models as baichuan2_13b_chat_model  # need torch 2.1
    # from .models.hf_internlm.hf_internlm2_chat_7b import models as hf_internlm2_chat_7b_model
    # from .models.hf_internlm.hf_internlm2_chat_20b import models as hf_internlm2_chat_20b_model
    # from .models.yi.hf_yi_6b_chat import models as yi_6b_chat_model
    # from .models.yi.hf_yi_34b_chat import models as yi_34b_chat_model
    # from .models.deepseek.hf_deepseek_7b_chat import models as deepseek_7b_chat_model
    # from .models.deepseek.hf_deepseek_67b_chat import models as deepseek_67b_chat_model
    # from .models.qwen.hf_qwen_7b_chat import models as qwen_7b_chat_model
    # from .models.qwen.hf_qwen_14b_chat import models as qwen_14b_chat_model
    # from .models.qwen.hf_qwen_72b_chat import models as qwen_72b_chat_model
    # <<<<<<------ https://arxiv.org/abs/2403.14112

    # from .models.openai.gpt_3_5_turbo_0125 import models as gpt_3_5_turbo_0125_model
    # from .models.openai.gpt_4o_2024_05_13 import models as gpt_4o_2024_05_13_model
    # from .models.gemini.gemini_1_5_flash import models as gemini_1_5_flash_model
    # from .models.gemini.gemini_1_5_pro import models as gemini_1_5_pro_model

    # from .models.hf_llama.lmdeploy_llama3_8b_instruct import models as lmdeploy_llama3_8b_instruct_model
    # from .models.hf_llama.lmdeploy_llama3_70b_instruct import models as lmdeploy_llama3_70b_instruct_model

    # from .models.hf_internlm.lmdeploy_internlm2_chat_1_8b import models as lmdeploy_internlm2_chat_1_8b_model
    # from .models.hf_internlm.lmdeploy_internlm2_chat_7b import models as lmdeploy_internlm2_chat_7b_model
    # from .models.hf_internlm.lmdeploy_internlm2_chat_20b import models as lmdeploy_internlm2_chat_20b_model

    # from .models.yi.hf_yi_1_5_6b_chat import models as yi_1_5_6b_chat_model
    # from .models.yi.hf_yi_1_5_34b_chat import models as yi_1_5_34b_chat_model

    # from .models.deepseek.hf_deepseek_v2_chat import models as deepseek_v2_chat_model

    # from .models.qwen.hf_qwen1_5_1_8b_chat import models as qwen1_5_1_8b_chat_model
    # from .models.qwen.hf_qwen1_5_7b_chat import models as qwen1_5_7b_chat_model
    # from .models.qwen.hf_qwen1_5_14b_chat import models as qwen1_5_14b_chat_model
    # from .models.qwen.hf_qwen1_5_72b_chat import models as qwen1_5_72b_chat_model

    from .summarizers.charm_reason import summarizer

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
work_dir = './outputs/CHARM_rea/chat/'

# dataset                                                        version    metric         mode    internlm2-chat-7b-turbomind
# -------------------------------------------------------------  ---------  -------------  ------  -----------------------------
# charm-reason-Direct                                               -          naive_average  gen     49.51
# charm-reason-ZH-CoT                                               -          naive_average  gen     61.33
# charm-reason-EN-CoT                                               -          naive_average  gen     54.55
# charm-reason-XLT                                                  -          naive_average  gen     58.46
# charm-reason-Translate-EN                                         -          naive_average  gen     56.15
#                                                                -          -              -       -
# charm-reason-Chinese_Direct                                       -          naive_average  gen     47.14
# charm-reason-Chinese_ZH-CoT                                       -          naive_average  gen     58.40
# charm-reason-Chinese_EN-CoT                                       -          naive_average  gen     48.31
# charm-reason-Chinese_XLT                                          -          naive_average  gen     53.57
# charm-reason-Chinese_Translate-EN                                 -          naive_average  gen     48.21
# charm-reason-Global_Direct                                        -          naive_average  gen     51.88
# charm-reason-Global_ZH-CoT                                        -          naive_average  gen     64.26
# charm-reason-Global_EN-CoT                                        -          naive_average  gen     60.79
# charm-reason-Global_XLT                                           -          naive_average  gen     63.36
# charm-reason-Global_Translate-EN                                  -          naive_average  gen     64.10
