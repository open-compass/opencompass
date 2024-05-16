from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.models import OpenAI
from opencompass.lagent.actions.ipython_interpreter import IPythonInterpreter
from opencompass.lagent.agents.react import CIReAct
from opencompass.models.lagent import CodeAgent
from lagent.agents.react import ReActProtocol

with read_base():
    from .datasets.CIBench.CIBench_gen_eb42f9 import cibench_datasets as datasets

FORCE_STOP_PROMPT_EN = """You should directly give results based on history information."""

FEWSHOT_INSTRUCTION = """\
You are an assistant who can utilize external tools.
{tool_description}
To use a tool, please response with the following format:
```
{thought} Think what you need to solve, do you need to use tools?
{action} The tool name, should be one of [{action_names}].
{action_input} The input to the tool that you want to use.
```
The tool will give you response after your response using the following format:
```
{response} the results after call the tool.
```
Therefore DO NOT generate tool response by yourself.

Also please follow the guidelines:
1. Always use code interpreter to solve the problem.
2. The generated codes should always in a markdown code block format.
3. The generated codes will be executed in an ipython manner and the results will be cached.
4. Your responded code should always be simple and only solves the problem in current step.

Begin!
"""

models = [
    dict(
        abbr='gpt-3.5-turbo',
        type=CodeAgent,
        agent_type=CIReAct,
        mutli_rounds=True,
        max_turn=3,
        llm=dict(
            type=OpenAI,
            path='gpt-3.5-turbo',
            key='ENV',
            query_per_second=1,
            max_seq_len=4096,
        ),
        actions=[
            dict(
                type=IPythonInterpreter,
                description=
                '''It can run Python code in a manner as jupyter notebook. The code must be a valid code that contains only python method.
'''),
        ],
        protocol=dict(
            type=ReActProtocol,
            call_protocol=FEWSHOT_INSTRUCTION,
            force_stop=FORCE_STOP_PROMPT_EN,
            action=dict(role='ACTION', begin='Tool:', end='\n'),
            action_input=dict(role='ARGS', begin='Tool Input:', end='\n'),
            response=dict(role='RESPONSE', begin='Tool Response:', end='\n'),
            finish=dict(role='FINISH', begin='Final Answer:', end='\n'),
        ),
        batch_size=8,
    ),
]


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=50, gen_task_coef=1),
    runner=dict(
        type=SlurmRunner, max_num_workers=8, retry=2,
        task=dict(type=OpenICLInferTask)),
)
