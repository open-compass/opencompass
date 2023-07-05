from mmengine.config import read_base

with read_base():
    from ..piqa.piqa_gen_8287ae import piqa_datasets
    from ..nq.nq_gen_a6ffca import nq_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
