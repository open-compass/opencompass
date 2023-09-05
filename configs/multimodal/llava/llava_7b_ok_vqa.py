from opencompass.multimodal.models.llava import LLaVAVQAPromptConstructor, LLaVABasePostProcessor

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
    dict(
        type='mmpretrain.PackInputs',
        algorithm_keys=['question', 'gt_answer', 'gt_answer_weight'],
        meta_keys=['question_id', 'image_id'],
    )
]

dataset = dict(
    type='mmpretrain.COCOVQA',
    data_root='data/okvqa',
    question_file='annotations/OpenEnded_mscoco_val2014_questions.json',
    ann_file='annotations/mscoco_val2014_annotations.json',
    pipeline=val_pipeline,
    data_prefix='images/val2014',
)

llava_okvqa_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dataset,
    collate_fn=dict(type='pseudo_collate'),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# model settings
llava_okvqa_model = dict(
    type='llava',
    model_path='/path/to/llava',
    prompt_constructor=dict(type=LLaVAVQAPromptConstructor),
    post_processor=dict(type=LLaVABasePostProcessor)
)  # noqa

# evaluation settings
llava_okvqa_evaluator = [dict(type='mmpretrain.VQAAcc')]


