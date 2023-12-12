from mmengine.config import read_base

with read_base():
    from .models.rwkv.rwkv5_3b import models
    from .datasets.collections.base_medium_llama import datasets
    from .summarizers.leaderboard import summarizer
