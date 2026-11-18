from mmengine.config import read_base

with read_base():
    from .rpcbench_open_llmjudge_rawprompt_gen import \
        rpcbench_open_datasets  # noqa: F401, F403
    from .rpcbench_cv_llmjudge_rawprompt_gen import \
        rpcbench_cv_datasets  # noqa: F401, F403

rpcbench_datasets = rpcbench_open_datasets + rpcbench_cv_datasets
