from mmengine.config import read_base

with read_base():
    from autotest.infer.config import \
        concurrent_infer as infer  # noqa: F401, E501
    from autotest.infer.config import models  # noqa: F401, E501
    from opencompass.configs.datasets.babilong.babilong_0k_gen import \
        babiLong_0k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.babilong.babilong_4k_gen import \
        babiLong_4k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.babilong.babilong_16k_gen import \
        babiLong_16k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.babilong.babilong_32k_gen import \
        babiLong_32k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.babilong.babilong_128k_gen import \
        babiLong_128k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.babilong.babilong_256k_gen import \
        babiLong_256k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.longbenchv2.longbenchv2_gen import \
        LongBenchv2_datasets as LongBenchv2_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.needlebench.needlebench_8k.needlebench_8k import \
        needlebench_datasets as needlebench_8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.needlebench.needlebench_32k.needlebench_32k import \
        needlebench_datasets as needlebench_32k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.needlebench.needlebench_128k.needlebench_128k import \
        needlebench_datasets as needlebench_128k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ruler.ruler_8k_gen import \
        ruler_datasets as ruler_8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ruler.ruler_32k_gen import \
        ruler_datasets as ruler_32k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ruler.ruler_64k_gen import \
        ruler_datasets as ruler_64k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ruler.ruler_128k_gen import \
        ruler_datasets as ruler_128k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ruler.ruler_256k_gen import \
        ruler_datasets as ruler_256k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ruler.ruler_512k_gen import \
        ruler_datasets as ruler_512k_datasets  # noqa: F401, E501

datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')),
    [],
)
