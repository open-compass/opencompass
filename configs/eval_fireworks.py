from mmengine.config import read_base
from opencompass.models import Fireworks
from opencompass.partitioners import NaivePartitioner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from .datasets.CLUE_cmnli.CLUE_cmnli_gen import cmnli_datasets
    from .datasets.CLUE_ocnli.CLUE_ocnli_gen import ocnli_datasets
    from .datasets.FewCLUE_ocnli_fc.FewCLUE_ocnli_fc_gen import ocnli_fc_datasets
    from .datasets.SuperGLUE_AX_b.SuperGLUE_AX_b_gen import AX_b_datasets
    from .datasets.SuperGLUE_AX_g.SuperGLUE_AX_g_gen import AX_g_datasets
    from .datasets.SuperGLUE_CB.SuperGLUE_CB_gen import CB_datasets
    from .datasets.SuperGLUE_RTE.SuperGLUE_RTE_gen import RTE_datasets
    from .datasets.anli.anli_gen import anli_datasets
datasets = [*CB_datasets]

models = [
    dict(abbr='mistral-7b',
        type=Fireworks, path='accounts/fireworks/models/mistral-7b',
        key='ENV',
        query_per_second=1,
        max_out_len=2048, max_seq_len=2048, batch_size=8),
]

work_dir = "outputs/api_mistral_7b/"