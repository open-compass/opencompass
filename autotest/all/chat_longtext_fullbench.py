from mmengine.config import read_base

with read_base():
    from autotest.all.config import \
        concurrent_infer as infer  # noqa: F401, E501
    from autotest.all.config import models  # noqa: F401, E501
    from opencompass.configs.datasets.babilong.babilong_256k_gen import \
        babiLong_256k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.longbenchv2.longbenchv2_gen import \
        LongBenchv2_datasets as LongBenchv2_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.needlebench.needlebench_32k.needlebench_32k import \
        needlebench_datasets as needlebench_32k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ruler.ruler_8k_gen import \
        ruler_datasets as ruler_8k_datasets  # noqa: F401, E501

datasets = sum(
    ([v[0]] if v else []
     for k, v in locals().items() if k.endswith('_datasets')),
    [],
)

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:1]'
