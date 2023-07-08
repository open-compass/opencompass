# dataloader settings
val_pipeline = [
    dict(type='mmpretrain.torchvision/Resize',
         size=(224, 224),
         interpolation=3),
    dict(type='mmpretrain.torchvision/ToTensor'),
    dict(type='mmpretrain.torchvision/Normalize',
         mean=(0.48145466, 0.4578275, 0.40821073),
         std=(0.26862954, 0.26130258, 0.27577711)),
    dict(type='mmpretrain.PackInputs',
         algorithm_keys=[
             'question', 'answer', 'options', 'category', 'l2-category',
             'context', 'index', 'options_dict'
         ])
]

dataset = dict(type='opencompass.OmniMMBenchDataset',
               data_file='data/mm_benchmark/mmagi_v030_full_inferin.tsv',
               pipeline=val_pipeline)

dataloader = dict(batch_size=1,
                  num_workers=4,
                  dataset=dataset,
                  collate_fn=dict(type='pseudo_collate'),
                  sampler=dict(type='DefaultSampler', shuffle=False))

# model settings
model = dict(
    type='visualglm-omnimmbench',
    path='/mnt/petrelfs/share_data/yuanyike/visualglm-6b',  # optional
    max_new_tokens=50,
    num_beams=5,
    do_sample=False,
    repetition_penalty=1.0,
    length_penalty=-1.0,
)

# evaluation settings
evaluator = [
    dict(
        type='opencompass.DumpResults',
        save_path='work_dirs/visualglm-6b-omnimmbench.xlsx')
]