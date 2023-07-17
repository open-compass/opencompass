from mmengine.config import read_base

with read_base():
    from .datasets.piqa.piqa_ppl import piqa_datasets
    from .datasets.siqa.siqa_gen import siqa_datasets
    from .models.hf_llama_7b import models


datasets = [*piqa_datasets, *siqa_datasets]
