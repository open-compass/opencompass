from opencompass.multimodal.models.visualglm import (VisualGLMBasePostProcessor, VisualGLMMMBenchPromptConstructor)

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
             'question', 'options', 'category', 'l2-category', 'context',
             'index', 'options_dict'
         ])
]

dataset = dict(type='opencompass.MMBenchDataset',
               data_file='data/mmbench/mmbench_test_20230712.tsv',
               pipeline=val_pipeline)

visualglm_mmbench_dataloader = dict(batch_size=1,
                  num_workers=4,
                  dataset=dataset,
                  collate_fn=dict(type='pseudo_collate'),
                  sampler=dict(type='DefaultSampler', shuffle=False))

# model settings
visualglm_mmbench_model = dict(
    type='visualglm',
    pretrained_path='/path/to/visualglm',  # or Huggingface repo id
    prompt_constructor=dict(type=VisualGLMMMBenchPromptConstructor),
    post_processor=dict(type=VisualGLMBasePostProcessor),
    gen_kwargs=dict(max_new_tokens=50,num_beams=5,do_sample=False,repetition_penalty=1.0,length_penalty=-1.0)
)

# evaluation settings
visualglm_mmbench_evaluator = [
    dict(type='opencompass.DumpResults',
         save_path='work_dirs/visualglm-6b-mmbench.xlsx')
]
