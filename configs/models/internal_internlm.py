from opencompass.models.internal import InternLMwithModule


models = [
    dict(
        type=InternLMwithModule,
        abbr="further_llama_7B_cot_1000",
        path="hdd_new_model_weights:s3://hdd_new_model_weights/0831/feizhaoye/further_llama_7B_cot_2/1000",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
        tokenizer_type='llama',
        module_path="./train_internlm/",
        model_config="./train_internlm/configs/further_llama_7B_cot.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)
    ),
]
