from mmengine.config import read_base

with read_base():
    from .elbench_safety_judge import elbench_safety_datasets
    from .elbench_highlevel_omni_gen import elbench_highlevel_omni_datasets
    from .elbench_highlevel_edu_judge import elbench_highlevel_edu_datasets
    from .elbench_general_gen import elbench_general_datasets

# All ELBench subsets runnable from the public ELBench dataset
# (ZeroLoss-Lab/ELBench) via OpenCompass's single-turn pipeline.
elbench_datasets = (elbench_safety_datasets +
                    elbench_highlevel_omni_datasets +
                    elbench_highlevel_edu_datasets + elbench_general_datasets)
