from mmengine.config import read_base
from opencompass.models.huggingface import HuggingFaceCausalLM
from opencompass.partitioners import NaivePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .datasets.mgsm.mgsm_gen import mgsm_datasets

  


# Eval MSGM_datasets
datasets = [*mgsm_datasets]