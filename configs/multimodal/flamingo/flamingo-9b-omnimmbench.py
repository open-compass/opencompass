# dataloader settings
val_pipeline = [
    dict(type='mmpretrain.PILToNumpy'),
    dict(type='mmpretrain.ResizeEdge',
         scale=224,
         interpolation='bicubic',
         backend='pillow'),
    dict(type='CenterCrop', crop_size=(224, 224)),
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
    batch_size=1,
    num_workers=4,
    dataset=dataset,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    persistent_workers=True,
)

# model settings
model = dict(
    type='flamingo-mm-benchmark',
    data_preprocessor=dict(
        type='mmpretrain.MultiModalDataPreprocessor',
        mean=[122.770938, 116.7460125, 104.09373615],
        std=[68.5005327, 66.6321579, 70.32316305],
        to_rgb=True,
    ),
    tokenizer=dict(type='mmpretrain.LlamaTokenizer',
                   name_or_path='decapoda-research/llama-7b-hf'),
    vision_encoder=dict(
        type='mmpretrain.VisionTransformer',
        arch='l',
        patch_size=14,
        pre_norm=True,
        norm_cfg=dict(type='LN', eps=1e-5),
        layer_cfgs=dict(act_cfg=dict(type='mmpretrain.QuickGELU')),
        final_norm=False,
        out_type='raw',
        pretrained=  # noqa: E251
        '/mnt/petrelfs/share_data/zhaowangbo/vit-large-p14_clip-openai-pre_3rdparty_20230517-95e2af0b.pth',
    ),
    lang_encoder=dict(
        base=dict(type='mmpretrain.AutoModelForCausalLM',
                  name_or_path='decapoda-research/llama-7b-hf',
                  local_files_only=True),
        adapter=dict(type='mmpretrain.FlamingoLMAdapter',
                     vis_hidden_size=1024,
                     cross_attn_every_n_layers=4,
                     use_media_placement_augmentation=False),
    ),
    generation_cfg=dict(num_beams=3, max_new_tokens=20, length_penalty=-2.0),
)

# evaluation settings
evaluator = [
    dict(type='opencompass.DumpResults',
         save_path='work_dirs/9b-flamingo/9b-flamingo-omnimmbench.xlsx')
]

load_from = '/mnt/petrelfs/share_data/zhaowangbo/openflamingo-9b-adapter_20230505-554310c8.pth'  # noqa

