from opencompass.multimodal.models.mplug_owl import (
    MplugOwlMMBenchPostProcessor, MplugOwlMMBenchPromptConstructor)

# dataloader settings
val_pipeline = [
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
        algorithm_keys=[
            'question', 'answer', 'category', 'l2-category', 'context',
            'index', 'options_dict', 'options'
        ],
    ),
]

dataset = dict(type='opencompass.MMBenchDataset',
               data_file='data/mmbench/mmbench_test_20230712.tsv',
               pipeline=val_pipeline)

mplug_owl_mmbench_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dataset,
    collate_fn=dict(type='pseudo_collate'),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# model settings
mplug_owl_mmbench_model = dict(
    type='mplug_owl-7b',
    model_path='/mplug-owl-llama-7b-ft',
    prompt_constructor=dict(type=MplugOwlMMBenchPromptConstructor),
    post_processor=dict(type=MplugOwlMMBenchPostProcessor)
)  # noqa

# evaluation settings
mplug_owl_mmbench_evaluator = [
    dict(type='opencompass.DumpResults',
         save_path='work_dirs/mplug_owl-7b-mmagibench-v0.1.0.xlsx')
]
