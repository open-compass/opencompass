from opencompass.multimodal.models.minigpt_4 import (
    MiniGPT4COCOCaotionPromptConstructor,
    MiniGPT4COCOCaptionPostProcessor,
)

# dataloader settings
val_pipeline = [
    dict(type='mmpretrain.LoadImageFromFile'),
    dict(type='mmpretrain.ToPIL', to_rgb=True),
    dict(type='mmpretrain.torchvision/Resize',
         size=(384, 384),
         interpolation=3),
    dict(type='mmpretrain.torchvision/ToTensor'),
    dict(type='mmpretrain.torchvision/Normalize',
         mean=(0.48145466, 0.4578275, 0.40821073),
         std=(0.26862954, 0.26130258, 0.27577711)),
    dict(type='mmpretrain.PackInputs', algorithm_keys=['image_id'])
]

dataset = dict(type='mmpretrain.COCOCaption',
               data_root='data/coco',
               data_prefix=dict(img_path='images'),
               ann_file='annotations/coco_karpathy_val.json',
               pipeline=val_pipeline)

minigpt_4_coco_caption_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dataset,
    collate_fn=dict(type='pseudo_collate'),
    sampler=dict(type='DefaultSampler', shuffle=False))

# model settings
minigpt_4_coco_caption_model = dict(
    type='minigpt-4',
    low_resource=False,
    img_size=384,
    llama_model='/path/to/vicuna_weights_7b/',
    is_caption_task=True,
    prompt_constructor=dict(type=MiniGPT4COCOCaotionPromptConstructor,
                            image_prompt='###Human: <Img><ImageHere></Img>',
                            reply_prompt='###Assistant:'),
    post_processor=dict(type=MiniGPT4COCOCaptionPostProcessor))

# evaluation settings
minigpt_4_coco_caption_evaluator = [
    dict(
        type='mmpretrain.COCOCaption',
        ann_file='data/coco/annotations/coco_karpathy_val_gt.json',
    )  # noqa
]

minigpt_4_coco_caption_load_from = '/path/to/prerained_minigpt4_7b.pth'  # noqa
