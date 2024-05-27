from mmengine.config import read_base

with read_base():
    from .datasets.CHARM.charm_rea_gen_f8fca2 import charm_rea_datasets as datasets
    from .models.hf_internlm.lmdeploy_internlm2_chat_7b import models as lmdeploy_7b_chat_model

    # from models.openai.gpt_3_5_turbo_1106 import models as gpt_3_5_turbo_1106_model
    # from models.openai.gpt_4_1106_preview import models as gpt_4_1106_preview_model

    # from .models.chatglm.hf_chatglm3_6b_32k import models as chatglm3_6b_32k_model
    # from .models.yi.hf_yi_6b_chat import models as yi_6b_chat_model
    # from .models.hf_internlm.hf_internlm2_chat_7b import models as hf_internlm2_chat_7b_model
    # from .models.deepseek.hf_deepseek_7b_chat import models as deepseek_7b_chat_model
    # from .models.baichuan.hf_baichuan2_7b_chat import models as baichuan2_7b_chat_model  # need torch 2.1
    # from .models.hf_llama.hf_llama2_7b_chat import models as llama2_7b_chat_model
    # from .models.vicuna.hf_vicuna_7b_v15_16k import models as vicuna_7b_v15_16k_model

    # from .models.baichuan.hf_baichuan2_13b_chat import models as baichuan2_13b_chat_model  # need torch 2.1
    # from .models.hf_llama.hf_llama2_13b_chat import models as llama2_13b_chat_model
    # from .models.vicuna.hf_vicuna_13b_v15_16k import models as vicuna_13b_v15_16k_model
    # from .models.hf_internlm.hf_internlm2_chat_20b import models as hf_internlm2_chat_20b_model

    # from .models.yi.hf_yi_34b_chat import models as yi_34b_chat_model

    # from .models.deepseek.hf_deepseek_67b_chat import models as deepseek_67b_chat_model
    # from .models.hf_llama.hf_llama2_70b_chat import models as llama2_70b_chat_model

    # from .models.hf_llama.hf_llama3_8b_instruct import models as llama3_8b_instruct_model
    # from .models.hf_llama.hf_llama3_70b_instruct import models as llama3_70b_instruct_model

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
work_dir = './outputs/CHARM/chat/'
