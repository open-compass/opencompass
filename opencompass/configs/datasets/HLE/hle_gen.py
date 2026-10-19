from mmengine.config import read_base

with read_base():
    # Default use LLM as a judge
    from .hle_llmverify_gen_6ff468 import hle_datasets  # noqa: F401, F403
