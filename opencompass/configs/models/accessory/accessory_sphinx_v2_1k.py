from opencompass.models import LLaMA2AccessoryModel

# Please follow the LLaMA2-Accessory installation document
# https://llama2-accessory.readthedocs.io/en/latest/install.html
# to install LLaMA2-Accessory

models = [
    dict(
        abbr='Accessory_sphinx_v2_1k',
        type=LLaMA2AccessoryModel,

        additional_stop_symbols=['###'],  # for models tuned with chat template

        # <begin> kwargs for accessory.MetaModel.from_pretrained
        # download from https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/tree/main/finetune/mm/SPHINX/SPHINX-v2-1k  # noqa
        pretrained_path='path/to/sphinx_v2_1k',
        llama_type=None,  # None for automatic probe from pretrained_path
        llama_config=None,  # None for automatic probe from pretrained_path
        tokenizer_path=None,  # None for automatic probe from pretrained_path
        with_visual=False,  # currently only support single-modal evaluation
        max_seq_len=4096,
        quant=False,
        # <end>

        batch_size=2,
        # LLaMA2-Accessory needs num_gpus==num_procs
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]
