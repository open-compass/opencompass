from mmengine.config import read_base

with read_base():
    # Default use LLM as a judge
    from .seedbench_gen_44868b import seedbench_datasets  # noqa: F401, F403
