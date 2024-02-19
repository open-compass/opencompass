from mmengine.config import read_base

from opencompass.lagent.actions.ipython_interpreter import IPythonInterpreter
from opencompass.lagent.agents.react import CIReAct, ReActProtocol
from opencompass.models.lagent import CodeAgent
from opencompass.models.openai_api import OpenAI
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from .datasets.CIBench.CIBench_template_gen_e6b12a import \
        cibench_datasets as datasets

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

For example:

File url: `xxxx`
### Step 1. Load the dataset from the url into a pandas DataFrame named `df`.

{thought} We should use `pandas` to solve this step.
{action} IPythonInterpreter
{action_input} ```python
import pandas as pd
url = "xxxx"
data = pd.read_csv(url)
```
{response} The code is succeed without any outputs.

Let us begin from here!
"""

IPYTHON_INTERPRETER_DESCRIPTION = '''\
It can run Python code in a manner as jupyter notebook. The code must be a valid code that contains only python method.'''

models = [
    dict(
        abbr='gpt-3.5-code',
        type=CodeAgent,
        agent_type=CIReAct,
        max_turn=3,
        llm=dict(
            type=OpenAI,
            path='gpt-3.5-turbo',
            key='ENV',
            query_per_second=1,
            max_seq_len=4096,
        ),
        actions=[
            dict(type=IPythonInterpreter,
                 description=IPYTHON_INTERPRETER_DESCRIPTION,
                 user_data_dir='./data/cibench_dataset/datasources')
        ],
        protocol=dict(
            type=ReActProtocol,
            call_protocol=FEWSHOT_INSTRUCTION,
            force_stop=FORCE_STOP_PROMPT_EN,
            finish=dict(role='FINISH', begin='Final Answer:', end='\n'),
        ),
        batch_size=1,
        use_system_role=False, # use `user` role instead of system role
        first_system_role=False, # use `user` role of the first instruction prompt
        merge_adjacent_role=True, # merge adjacent same user content
    ),
]


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask)),
)
