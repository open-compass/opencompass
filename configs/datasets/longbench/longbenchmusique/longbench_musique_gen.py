from mmengine.config import read_base

with read_base():
    from .longbench_musique_gen_6b3efc import LongBench_musique_datasets  # noqa: F401, F403
