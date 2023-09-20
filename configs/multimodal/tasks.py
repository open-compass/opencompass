from mmengine.config import read_base

with read_base():
    from .mplug_owl.mplug_owl_7b_coco_caption import (
        mplug_owl_coco_caption_dataloader,
        mplug_owl_coco_caption_model,
        mplug_owl_coco_caption_evaluator,
    )

models = [mplug_owl_coco_caption_model]
datasets = [mplug_owl_coco_caption_dataloader]
evaluators = [mplug_owl_coco_caption_evaluator]
load_froms = [None]

num_gpus = 4
num_procs = 4
launcher = 'pytorch'
