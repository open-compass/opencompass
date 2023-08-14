from opencompass.multimodal.models.minigpt_4 import (MiniGPT4PostProcessorMME, MiniGPT4MMEPromptConstructor)

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
    dict(type='mmpretrain.PackInputs',
         algorithm_keys=[
             'question', 'answer', 'task'
         ])
]

dataset = dict(type='opencompass.MMEDataset',
               data_dir='/path/to/MME',
               pipeline=val_pipeline)

minigpt_4_mme_dataloader = dict(batch_size=1,
                            num_workers=4,
                            dataset=dataset,
                            collate_fn=dict(type='pseudo_collate'),
                            sampler=dict(type='DefaultSampler', shuffle=False))

# model settings
minigpt_4_model = dict(
    type='minigpt-4-mmbench',
    low_resource=False,
    llama_model='/path/to/vicuna/',
    prompt_constructor=dict(type=MiniGPT4MMEPromptConstructor,
                            image_prompt='###Human: <Img><ImageHere></Img>',
                            reply_prompt='###Assistant:'),
    post_processor=dict(type=MiniGPT4PostProcessorMME))

# evaluation settings
minigpt_4_evaluator = [
    dict(type='opencompass.MMEMetric')
]

minigpt_4_load_from = '/path/to/prerained_minigpt4_7b.pth'  # noqa
