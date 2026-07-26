from mmengine.config import read_base

with read_base():
    from .helium_market_resolution_gen import helium_market_resolution_datasets  # noqa: F401, F403
