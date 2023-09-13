from opencompass.multimodal.models import (MiniGPT4ScienceQAPromptConstructor,
                                           MiniGPT4ScienceQAPostProcessor)

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
             'question', 'gt_answer', 'choices', 'hint', 'lecture', 'solution', 'has_image'
         ])
]

dataset = dict(type='mmpretrain.ScienceQA',
               data_root='./data/scienceqa',
               split='val',
               split_file='pid_splits.json',
               ann_file='problems.json',
               image_only=True,
               data_prefix=dict(img_path='val'),
               pipeline=val_pipeline)

minigpt_4_scienceqa_dataloader = dict(batch_size=1,
                                      num_workers=4,
                                      dataset=dataset,
                                      collate_fn=dict(type='pseudo_collate'),
                                      sampler=dict(type='DefaultSampler',
                                                   shuffle=False))

# model settings
minigpt_4_scienceqa_model = dict(
    type='minigpt-4',
    low_resource=False,
    img_size=224,
    max_length=10,
    llama_model='/path/to/vicuna_weights_7b/',
    prompt_constructor=dict(type=MiniGPT4ScienceQAPromptConstructor,
                            image_prompt='###Human: <Img><ImageHere></Img>',
                            reply_prompt='###Assistant:'),
    post_processor=dict(type=MiniGPT4ScienceQAPostProcessor))

# evaluation settings
minigpt_4_scienceqa_evaluator = [dict(type='mmpretrain.ScienceQAMetric')]

minigpt_4_scienceqa_load_from = '/path/to/prerained_minigpt4_7b.pth'  # noqa
