from copy import deepcopy
from mmengine.config import read_base
from opencompass.models.lagent import LagentAgent
from lagent import ReAct
from lagent.agents.react import ReActProtocol
from opencompass.models.lagent import CodeAgent
from opencompass.lagent.actions.python_interpreter import PythonInterpreter
from opencompass.lagent.actions.ipython_interpreter import IPythonInterpreter
from opencompass.lagent.agents.react import CIReAct
from opencompass.models import HuggingFaceCausalLM
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.partitioners import NaivePartitioner

with read_base():
    # Note that it might occur cuda OOM error for hf model
    from .models.hf_llama.lmdeploy_llama3_8b_instruct import models as lmdeploy_llama3_8b_instruct_model

    from .summarizers.cibench import summarizer
    from .datasets.CIBench.CIBench_template_gen_e6b12a import cibench_datasets as cibench_datasets_template
    from .datasets.CIBench.CIBench_generation_gen_8ab0dc import cibench_datasets as cibench_datasets_generation
    # Oracle mode for analysis
    # from .datasets.CIBench.CIBench_template_oracle_gen_fecda1 import cibench_datasets as cibench_datasets_template_oracle
    # from .datasets.CIBench.CIBench_generation_oracle_gen_c4a7c1 import cibench_datasets as cibench_datasets_generation_oracle

datasets = []
datasets += cibench_datasets_template
datasets += cibench_datasets_generation
# datasets += cibench_datasets_template_oracle
# datasets += cibench_datasets_generation_oracle

_origin_models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

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



actions=[dict(type=IPythonInterpreter, user_data_dir='./data/cibench_dataset/datasources',
                 description=IPYTHON_INTERPRETER_DESCRIPTION)]
protocol=dict(
            type=ReActProtocol,
            call_protocol=FEWSHOT_INSTRUCTION,
            force_stop=FORCE_STOP_PROMPT_EN,
            finish=dict(role='FINISH', begin='Final Answer:', end='\n'),
        )

work_dir = './outputs/cibench/'

_agent_models = []
for m in _origin_models:
    m = deepcopy(m)
    if 'meta_template' in m and 'round' in m['meta_template']:
        round = m['meta_template']['round']
        if all(r['role'].upper() != 'SYSTEM' for r in round):  # no system round
            if not any('api_role' in r for r in round):
                m['meta_template']['round'].append(dict(role='system', begin='System response:', end='\n'))
            else:
                m['meta_template']['round'].append(dict(role='system', api_role='SYSTEM'))
            print(f'WARNING: adding SYSTEM round in meta_template for {m.get("abbr", None)}')
    _agent_models.append(m)

protocol=dict(
    type=ReActProtocol,
    call_protocol=FEWSHOT_INSTRUCTION,
    force_stop=FORCE_STOP_PROMPT_EN,
    finish=dict(role='FINISH', begin='Final Answer:', end='\n'),
)

models = []
for m in _agent_models:
    m = deepcopy(m)
    origin_abbr = m.pop('abbr')
    abbr = origin_abbr
    m.pop('batch_size', None)
    m.pop('max_out_len', None)
    m.pop('max_seq_len', None)
    run_cfg = m.pop('run_cfg', {})

    agent_model = dict(
        abbr=abbr,
        summarizer_abbr=origin_abbr,
        type=CodeAgent,
        agent_type=CIReAct,
        max_turn=3,
        llm=m,
        actions=[dict(type=IPythonInterpreter, user_data_dir='./data/cibench_dataset/datasources', description=IPYTHON_INTERPRETER_DESCRIPTION)],
        protocol=protocol,
        batch_size=1,
        run_cfg=run_cfg,
    )
    models.append(agent_model)

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=4,
        task=dict(type=OpenICLInferTask)),
)
