from opencompass.multimodal.models.minigpt_4 import MiniGPT4SEEDBenchPromptConstructor  # noqa

# dataloader settings
image_pipeline = [
    dict(type='mmpretrain.torchvision/Resize',
         size=(224, 224),
         interpolation=3),
    dict(type='mmpretrain.torchvision/ToTensor'),
    dict(type='mmpretrain.torchvision/Normalize',
         mean=(0.48145466, 0.4578275, 0.40821073),
         std=(0.26862954, 0.26130258, 0.27577711)),
    dict(type='mmpretrain.PackInputs',
         algorithm_keys=[
             'question', 'answer', 'choices', 'data_type', 'question_type_id',
             'index', 'data_path', 'question_id'
         ])
]
video_pipeline = [
    dict(type='mmaction.Resize', scale=(224, 224), interpolation='bicubic'),
    dict(type='mmaction.CenterCrop', crop_size=224),
    dict(type='Normalize',
         mean=(0.48145466, 0.4578275, 0.40821073),
         std=(0.26862954, 0.26130258, 0.27577711)),
    dict(type='mmpretrain.PackInputs',
         algorithm_keys=[
             'question', 'answer', 'choices', 'data_type', 'question_type_id',
             'index', 'data_path', 'question_id'
         ])
]

dataset = dict(
    type='opencompass.SEEDBenchDataset',
    ann_file='data/seedbench/SEED-Bench.json',
    cc3m_path='data/seedbench/SEED-Bench-image',
    sthv2_path='data/seedbench/sthv2/videos',
    epic_kitchens_path='data/seedbench/3h91syskeag572hl6tvuovwv4d/videos/test',
    breakfast_path='data/seedbench/BreakfastII_15fps_qvga_sync',
    image_pipeline=image_pipeline,
    video_pipeline=video_pipeline,
    only_image=True)

minigpt_4_seedbench_dataloader = dict(batch_size=1,
                                      num_workers=4,
                                      dataset=dataset,
                                      collate_fn=dict(type='pseudo_collate'),
                                      sampler=dict(type='DefaultSampler',
                                                   shuffle=False))

# model settings
minigpt_4_seedbench_model = dict(
    type='minigpt-4',
    low_resource=False,
    llama_model='/path/to/vicuna/',
    prompt_constructor=dict(type=MiniGPT4SEEDBenchPromptConstructor,
                            image_prompt='###Human: <Img><ImageHere></Img>',
                            reply_prompt='###Assistant:'),
    post_processor=None,
    mode='loss')

# evaluation settings
minigpt_4_seedbench_evaluator = [dict(type='opencompass.SEEDBenchAcc')]

minigpt_4_load_from = '/path/to/prerained_minigpt4_7b.pth'
