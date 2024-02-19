from opencompass.models import Mixtral

# Please follow the instruction in https://github.com/open-compass/MixtralKit
# to download the model weights and install the requirements


models = [
    dict(
        abbr='mixtral-8x7b-32k',
        type=Mixtral,
        path='./models/mixtral/mixtral-8x7b-32kseqlen',
        tokenizer_path='./models/mixtral/mixtral-8x7b-32kseqlen/tokenizer.model',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        num_gpus=2,
        run_cfg=dict(num_gpus=2, num_procs=1),
    ),
]
