from mmengine.config import read_base

with read_base():
    from .ruler_1m_gen import ruler_datasets as ruler_1m_ds
    from .ruler_4k_gen import ruler_datasets as ruler_4k_ds
    from .ruler_8k_gen import ruler_datasets as ruler_8k_ds
    from .ruler_16k_gen import ruler_datasets as ruler_16k_ds
    from .ruler_32k_gen import ruler_datasets as ruler_32k_ds
    from .ruler_64k_gen import ruler_datasets as ruler_64k_ds
    from .ruler_128k_gen import ruler_datasets as ruler_128k_ds
    from .ruler_256k_gen import ruler_datasets as ruler_256k_ds
    from .ruler_512k_gen import ruler_datasets as ruler_512k_ds

ruler_combined_datasets = sum((v for k, v in locals().items() if k.endswith('_ds')), [])
