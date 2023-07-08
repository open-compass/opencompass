# dataloader settings
val_pipeline = [
    dict(type='mmpretrain.PILToNumpy'),
    dict(type='mmpretrain.ResizeEdge',
         scale=224,
         interpolation='bicubic',
         backend='pillow'),
    dict(type='CenterCrop', crop_size=(224, 224)),
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

dataset = dict(type='opencompass.OmniMMBenchDataset',
               data_file='data/mm_benchmark/mmagi_v030_full_inferin.tsv',
               pipeline=val_pipeline)

dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dataset,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    persistent_workers=True,
)

# model settings
model = dict(
    type='openflamingov2-omnimmbench',
    ckpt_path='/mnt/petrelfs/share_data/yuanyike/'
    'openflamingo/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt',
    # model init params
    clip_vision_encoder_path='ViT-L-14',
    clip_vision_encoder_pretrained='openai',
    lang_encoder_path='anas-awadalla/mpt-7b',
    tokenizer_path='anas-awadalla/mpt-7b',
    cross_attn_every_n_layers=4,
)

# evaluation settings
evaluator = [
    dict(
        type='opencompass.DumpResults',
        save_path=  # noqa: E251
        'work_dirs/openflamingov2-9b-omnimmbench.xlsx'
    )
]
