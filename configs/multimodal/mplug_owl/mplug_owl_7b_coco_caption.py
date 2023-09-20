from opencompass.multimodal.models.mplug_owl import (
    MplugOwlCOCOCaptionPromptConstructor, MplugOwlBasePostProcessor)

# dataloader settings
val_pipeline = [
    dict(type='mmpretrain.LoadImageFromFile'),
    dict(type='mmpretrain.ToPIL', to_rgb=True),
    dict(type='mmpretrain.torchvision/Resize',
         size=(224, 224),
         interpolation=3),
    dict(type='mmpretrain.torchvision/ToTensor'),
    dict(
        type='mmpretrain.torchvision/Normalize',
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
    dict(type='mmpretrain.PackInputs', algorithm_keys=['image_id']),
]

dataset = dict(type='mmpretrain.COCOCaption',
               data_root='data/coco',
               data_prefix=dict(img_path='images'),
               ann_file='annotations/coco_karpathy_val.json',
               pipeline=val_pipeline)

mplug_owl_coco_caption_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dataset,
    collate_fn=dict(type='pseudo_collate'),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# model settings
mplug_owl_coco_caption_model = dict(
    type='mplug_owl',
    model_path=
    '/mnt/petrelfs/share_data/liuyuan/llm_weights/mplug-owl-llama-7b-ft',
    is_caption_task=True,
    prompt_constructor=dict(type=MplugOwlCOCOCaptionPromptConstructor),
    post_processor=dict(type=MplugOwlBasePostProcessor))  # noqa

# evaluation settings
mplug_owl_coco_caption_evaluator = [
    dict(
        type='mmpretrain.COCOCaption',
        ann_file='data/coco/annotations/coco_karpathy_val_gt.json',
    )
]
