from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.collections.base_medium_llama import \
        datasets
    from opencompass.configs.models.rwkv.rwkv5_3b import models
    from opencompass.configs.summarizers.leaderboard import summarizer
