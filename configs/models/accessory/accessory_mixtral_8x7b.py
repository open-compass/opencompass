from opencompass.models import LLaMA2AccessoryModel

# Please follow the LLaMA2-Accessory installation document
# https://llama2-accessory.readthedocs.io/en/latest/install.html
# to install LLaMA2-Accessory

models = [
    dict(
        abbr="Accessory_mixtral_8x7b",
        type=LLaMA2AccessoryModel,

        # additional_stop_symbols=["###"],  # for models tuned with chat template
        additional_stop_symbols=[],

        # <begin> kwargs for accessory.MetaModel.from_pretrained
        # download from https://huggingface.co/Alpha-VLLM/MoE-Mixtral-7B-8Expert/tree/main/converted_sparse
        # see https://llama2-accessory.readthedocs.io/en/latest/projects/mixtral-8x7b.html for more details
        pretrained_path="path/to/MoE-Mixtral-7B-8Expert/converted_sparse",
        llama_type=None,  # set to None to automatically probe from pretrain_path
        llama_config=None,  # set to None to automatically probe from pretrain_path
        tokenizer_path=None,  # set to None to automatically probe from pretrain_path
        with_visual=False,
        max_seq_len=4096,
        quant=False,
        # <end>

        batch_size=2,
        run_cfg=dict(num_gpus=2, num_procs=2),  # LLaMA2-Accessory needs num_gpus==num_procs
    ),
]
