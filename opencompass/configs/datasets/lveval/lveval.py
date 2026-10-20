from mmengine.config import read_base

with read_base():
    from .lvevalcmrc_mixup.lveval_cmrc_mixup_gen import (
        LVEval_cmrc_mixup_datasets,
    )
    from .lvevaldureader_mixup.lveval_dureader_mixup_gen import (
        LVEval_dureader_mixup_datasets,
    )
    from .lvevalfactrecall_en.lveval_factrecall_en_gen import (
        LVEval_factrecall_en_datasets,
    )
    from .lvevalfactrecall_zh.lveval_factrecall_zh_gen import (
        LVEval_factrecall_zh_datasets,
    )
    from .lvevalhotpotwikiqa_mixup.lveval_hotpotwikiqa_mixup_gen import (
        LVEval_hotpotwikiqa_mixup_datasets,
    )
    from .lvevallic_mixup.lveval_lic_mixup_gen import LVEval_lic_mixup_datasets
    from .lvevalloogle_CR_mixup.lveval_loogle_CR_mixup_gen import (
        LVEval_loogle_CR_mixup_datasets,
    )
    from .lvevalloogle_MIR_mixup.lveval_loogle_MIR_mixup_gen import (
        LVEval_loogle_MIR_mixup_datasets,
    )
    from .lvevalloogle_SD_mixup.lveval_loogle_SD_mixup_gen import (
        LVEval_loogle_SD_mixup_datasets,
    )
    from .lvevalmultifieldqa_en_mixup.lveval_multifieldqa_en_mixup_gen import (
        LVEval_multifieldqa_en_mixup_datasets,
    )
    from .lvevalmultifieldqa_zh_mixup.lveval_multifieldqa_zh_mixup_gen import (
        LVEval_multifieldqa_zh_mixup_datasets,
    )

LVEval_datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')), []
)
