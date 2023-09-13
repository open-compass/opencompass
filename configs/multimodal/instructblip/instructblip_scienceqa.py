from opencompass.multimodal.models.instructblip import (
    InstructBlipScienceQAPromptConstructor,
    InstructBlipScienceQAPostProcessor,
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

instruct_blip_scienceqa_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dataset,
    collate_fn=dict(type='pseudo_collate'),
    sampler=dict(type='DefaultSampler', shuffle=False))

# model settings
instruct_blip_scienceqa_model = dict(
    type='blip2-vicuna-instruct',
    prompt_constructor=dict(type=InstructBlipScienceQAPromptConstructor),
    post_processor=dict(type=InstructBlipScienceQAPostProcessor),
    freeze_vit=True,
    low_resource=False,
    llm_model='/path/to/vicuna-7b/',
    max_output_txt_len=10,
)

# evaluation settings
instruct_blip_scienceqa_evaluator = [dict(type='mmpretrain.ScienceQAMetric')]

instruct_blip_load_from = '/path/to/instruct_blip_vicuna7b_trimmed.pth'
