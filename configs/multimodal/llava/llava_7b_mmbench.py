# dataloader settings
val_pipeline = [
    dict(type='mmpretrain.torchvision/Resize',
         size=(224, 224),
         interpolation=3),
    dict(type='mmpretrain.torchvision/ToTensor'),
    dict(
        type='mmpretrain.torchvision/Normalize',
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
    dict(
        type='mmpretrain.PackInputs',
        algorithm_keys=[
            'question', 'category', 'l2-category', 'context', 'index',
             'options_dict', 'options', 'split'
        ],
    ),
]

dataset = dict(type='opencompass.MMBenchDataset',
               data_file='data/mmbench/mmbench_test_20230712.tsv',
               pipeline=val_pipeline)

mmbench_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dataset,
    collate_fn=dict(type='pseudo_collate'),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# model settings
llava_model = dict(
    type='llava-7b-mmbench',
    model_path='/path/to/llava',
)  # noqa
"""
You can download llava's weights to your own path, remember you need to convert the delta weights to full model weights
following: https://github.com/haotian-liu/LLaVA/blob/main/README.md#llava-weights
"""

# evaluation settings
mmbench_evaluator = [
    dict(type='opencompass.DumpResults',
         save_path='work_dirs/llava-7b-mmbench.xlsx')
]
