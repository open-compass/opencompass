from mmengine.config import read_base

with read_base():
    # from .minigpt_4.minigpt_4_7b_mmbench import (minigpt_4_mmbench_dataloader,
    #                                              minigpt_4_mmbench_evaluator,
    #                                              minigpt_4_mmbench_load_from,
    #                                              minigpt_4_mmbench_model)
    # from .minigpt_4.minigpt_4_7b_coco_caption import (
    #     minigpt_4_coco_caption_dataloader, minigpt_4_coco_caption_evaluator,
    #     minigpt_4_coco_caption_load_from, minigpt_4_coco_caption_model)
    # from .minigpt_4.minigpt_4_7b_gqa import (minigpt_4_gqa_dataloader,
    #                                          minigpt_4_gqa_evaluator,
    #                                          minigpt_4_gqa_load_from,
    #                                          minigpt_4_gqa_model)
    from .minigpt_4.minigpt_4_7b_flickr30k import (
        minigpt_4_flickr30k_dataloader, minigpt_4_flickr30k_evaluator,
        minigpt_4_flickr30k_load_from, minigpt_4_flickr30k_model)

# models = [minigpt_4_mmbench_model]
# datasets = [minigpt_4_mmbench_dataloader]
# evaluators = [minigpt_4_mmbench_evaluator]
# load_froms = [minigpt_4_mmbench_load_from]
models = [minigpt_4_flickr30k_model]
datasets = [minigpt_4_flickr30k_dataloader]
evaluators = [minigpt_4_flickr30k_evaluator]
load_froms = [minigpt_4_flickr30k_load_from]

num_gpus = 8
num_procs = 8
launcher = 'pytorch'