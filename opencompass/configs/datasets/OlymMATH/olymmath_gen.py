from mmengine.config import read_base

with read_base():
    # Default use LLM as a judge
    from .olymmath_llmverify_gen_6ff468 import olymmath_datasets  # noqa: F401, F403
