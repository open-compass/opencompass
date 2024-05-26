from opencompass.models import LLaMA2AccessoryModel

# Please follow the LLaMA2-Accessory installation document
# https://llama2-accessory.readthedocs.io/en/latest/install.html
# to install LLaMA2-Accessory

models = [
    dict(
        abbr='Accessory_llama2_7b',
        type=LLaMA2AccessoryModel,

        # additional_stop_symbols=["###"],  # for models tuned with chat template  # noqa
        additional_stop_symbols=[],

        # <begin> kwargs for accessory.MetaModel.from_pretrained
        # download https://huggingface.co/meta-llama/Llama-2-7b/tree/main to
        # 'path/to/Llama-2-7b/', which should contain:
        #   - consolidated.00.pth
        #   - params.json
        #   - tokenizer.model
        pretrained_path='path/to/Llama-2-7b/',
        llama_type='llama',
        llama_config='path/to/Llama-2-7b/params.json',
        tokenizer_path='path/to/Llama-2-7b/tokenizer.model',
        with_visual=False,
        max_seq_len=4096,
        quant=False,
        # <end>

        batch_size=2,
        # LLaMA2-Accessory needs num_gpus==num_procs
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]
