from mmengine.config import read_base

with read_base():
    from .qasper_gen_db6413 import qasper_datasets  # noqa: F401, F403
