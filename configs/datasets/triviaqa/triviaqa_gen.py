from mmengine.config import read_base

with read_base():
    from .triviaqa_gen_2121ce import triviaqa_datasets  # noqa: F401, F403
