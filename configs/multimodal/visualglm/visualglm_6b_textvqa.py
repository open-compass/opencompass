from opencompass.multimodal.models.visualglm import (VisualGLMBasePostProcessor, VisualGLMVQAPromptConstructor)

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

dataset = dict(
    type='mmpretrain.TextVQA',
    data_root='data/textvqa',
    ann_file='annotations/TextVQA_0.5.1_val.json',
    pipeline=val_pipeline,
    data_prefix='images/train_images',
)

visualglm_textvqa_dataloader = dict(batch_size=1,
                  num_workers=4,
                  dataset=dataset,
                  collate_fn=dict(type='pseudo_collate'),
                  sampler=dict(type='DefaultSampler', shuffle=False))

# model settings
visualglm_textvqa_model = dict(
    type='visualglm',
    pretrained_path='/path/to/visualglm',  # or Huggingface repo id
    prompt_constructor=dict(type=VisualGLMVQAPromptConstructor),
    post_processor=dict(type=VisualGLMBasePostProcessor)
)

# evaluation settings
visualglm_textvqa_evaluator = [dict(type='mmpretrain.VQAAcc')]
