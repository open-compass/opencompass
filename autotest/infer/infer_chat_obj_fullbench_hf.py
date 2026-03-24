from mmengine.config import read_base

with read_base():
    from autotest.infer.config import infer  # noqa: F401, E501
    from autotest.infer.config import models  # noqa: F401, E501
    from opencompass.configs.datasets.eese.eese_llm_judge_gen import \
        eese_datasets  # noqa: F401, E501

datasets = sum(
    (v for k, v in locals().items()),
    [],
)
