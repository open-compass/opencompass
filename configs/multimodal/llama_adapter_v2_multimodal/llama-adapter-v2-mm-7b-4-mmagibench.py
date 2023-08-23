# dataloader settings
val_pipeline = [
    dict(type='mmpretrain.torchvision/Resize',
         size=(224, 224),
         interpolation=3),
    dict(type='mmpretrain.torchvision/ToTensor'),
    dict(type='mmpretrain.torchvision/Normalize',
         mean=(0.48145466, 0.4578275, 0.40821073),
         std=(0.26862954, 0.26130258, 0.27577711)),
    dict(type='mmpretrain.PackInputs',
         algorithm_keys=[
             'question', 'answer', 'options', 'category', 'l2-category',
             'index', 'context', 'options_dict'
         ])
]

dataset = dict(type='opencompass.MMBenchDataset',
               data_file='data/mmbench/mmbench_test_20230712.tsv',
               pipeline=val_pipeline)

dataloader = dict(batch_size=1,
                  num_workers=4,
                  dataset=dataset,
                  collate_fn=dict(type='pseudo_collate'),
                  sampler=dict(type='DefaultSampler', shuffle=False))

# model settings
model = dict(
    type='LLaMA-adapter-v2-mm-benchmark',
    llama_dir=  # noqa
    '/mnt/petrelfs/share_data/zhangyuanhan/llama_adapter_v2_multimodal',
)

# evaluation settings
evaluator = [
    dict(
        type='opencompass.DumpResults',
        save_path='work_dirs/llama-adapter-v2-multimodal-mmagibench-v0.1.0.xlsx'
    )
]

# load_from = '/mnt/cache/liuyuan/research/NLP/MiniGPT-4/minigpt4-7b/prerained_minigpt4_7b.pth'  # noqa
