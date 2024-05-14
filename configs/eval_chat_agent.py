from mmengine.config import read_base
from opencompass.models.openai_api import OpenAI
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.models.lagent import LagentAgent
from opencompass.lagent.actions.python_interpreter import PythonInterpreter
from lagent import ReAct
from lagent.agents.react import ReActProtocol

with read_base():
    from .datasets.gsm8k.gsm8k_agent_gen_be1606 import gsm8k_datasets
    from .datasets.math.math_agent_gen_af2293 import math_datasets
    from .datasets.MathBench.mathbench_agent_gen_568903 import mathbench_agent_datasets
    from .summarizers.math_agent import summarizer

datasets = []
datasets += gsm8k_datasets
datasets += math_datasets
datasets += mathbench_agent_datasets

system_prompt = """You are a helpful assistant which use tools to solve mathematical reasoning questions. The code must be a function, and the function name must be 'solution'. For mathematics, please use code tool to calculate. The example format is as follows:
```
def solution():
    variable_names_with_real_meaning = func(variable)
    return variable_names_with_real_meaning
```"""

protocol = dict(
    type=ReActProtocol,
    action=dict(role='ACTION', begin='Tool:', end='\n'),
    action_input=dict(role='ARGS', begin='Tool Input:', end='\n'),
    finish=dict(role='FINISH', begin='FinalAnswer:', end='\n'),
    call_protocol=system_prompt,
)

models = [
    dict(
        abbr='gpt-3.5-react',
        type=LagentAgent,
        agent_type=ReAct,
        max_turn=3,
        llm=dict(
            type=OpenAI,
            path='gpt-3.5-turbo',
            key='ENV',
            query_per_second=1,
            max_seq_len=4096,
        ),
        actions=[
            dict(type=PythonInterpreter),
        ],
        protocol=protocol,
        batch_size=1,
    ),
]

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask)),
)
