from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.models import OpenAI
from opencompass.models.lagent import CodeAgent
from opencompass.lagent.actions.python_interpreter import PythonInterpreter

PYTHON_INTERPRETER_DESCRIPTION = """\
It can run a Python code. The code must be a valid code that contains only python method.
"""

actions = [
    dict(
        type=PythonInterpreter,
        description=PYTHON_INTERPRETER_DESCRIPTION,
        answer_expr=None,
    )
]

with read_base():
    from .datasets.ds1000.ds1000_gen_5c4bec import ds1000_datasets as datasets

models = [
    dict(
        abbr='gpt-3.5-react',
        type=CodeAgent,
        llm=dict(
            type=OpenAI,
            path='gpt-3.5-turbo',
            key='ENV',
            query_per_second=1,
            max_seq_len=4096,
        ),
        actions=actions,
        batch_size=8),
]

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=40000),
    runner=dict(
        type=LocalRunner, max_num_workers=16,
        task=dict(type=OpenICLInferTask)),
)
