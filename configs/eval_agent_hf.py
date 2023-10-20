from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from .datasets.math.math_gen_66176f import math_datasets, math_example
    from .models.agent_template import model_template
    from .models.wizardcoder.hf_wizardcoder_python_13b import models as wizard_model

_task_keys = ['abbr', 'max_out_len', 'batch_size', 'run_cfg']
_task_kwargs = {}
for _key in _task_keys:
    if _key in wizard_model[0]:
        _task_kwargs[_key] = wizard_model[0].pop(_key)

datasets = math_datasets
models = [
    dict(
        llm=wizard_model[0],
        example=math_example,
        **model_template,
        **_task_kwargs),
]

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=40000),
    runner=dict(
        type=LocalRunner, max_num_workers=16,
        task=dict(type=OpenICLInferTask)),
)
