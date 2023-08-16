from opencompass.multimodal.models.minigpt_4 import (
    MiniGPT4VSRPromptConstructor,
    MiniGPT4VSRPostProcessor,
)

# dataloader settings
val_pipeline = [
    dict(type='mmpretrain.LoadImageFromFile'),
    dict(type='mmpretrain.ToPIL', to_rgb=True),
    dict(type='mmpretrain.torchvision/Resize',
         size=(224, 224),
         interpolation=3),
    dict(type='mmpretrain.torchvision/ToTensor'),
    dict(type='mmpretrain.torchvision/Normalize',
         mean=(0.48145466, 0.4578275, 0.40821073),
         std=(0.26862954, 0.26130258, 0.27577711)),
    dict(
        type='mmpretrain.PackInputs',
        algorithm_keys=['question', 'gt_answer', 'gt_answer_weight'],
        meta_keys=['question_id', 'image_id'],
    )
]

dataset = dict(type='mmpretrain.VSR',
               data_root='data/vsr/',
               data_prefix='images/',
               ann_file='annotations/test.json',
               pipeline=val_pipeline)

minigpt_4_vsr_dataloader = dict(batch_size=1,
                                num_workers=4,
                                dataset=dataset,
                                collate_fn=dict(type='pseudo_collate'),
                                sampler=dict(type='DefaultSampler',
                                             shuffle=False))

# model settings
minigpt_4_vsr_model = dict(
    type='minigpt-4',
    low_resource=True,
    img_size=224,
    max_length=10,
    llama_model='/path/to/vicuna-7b/',
    prompt_constructor=dict(type=MiniGPT4VSRPromptConstructor,
                            image_prompt='###Human: <Img><ImageHere></Img>',
                            reply_prompt='###Assistant:'),
    post_processor=dict(type=MiniGPT4VSRPostProcessor))

# evaluation settings
minigpt_4_vsr_evaluator = [dict(type='mmpretrain.GQAAcc')]

minigpt_4_vsr_load_from = '/path/to/prerained_minigpt4_7b.pth'  # noqa
