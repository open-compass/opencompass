from mmengine.config import read_base

with read_base():
    from .triviaqa_gen_cc3cbf import triviaqa_datasets  # noqa: F401, F403
