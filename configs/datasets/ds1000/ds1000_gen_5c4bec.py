from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer
from opencompass.datasets import DS1000Dataset_Interperter, DS1000InterpreterEvaluator

ds1000_example = """
In the following task, you should generate code with one assertion to testify the correctness of your code.

Example:

<HUMAN>Problem:
How do I get the dimensions of an array? For instance, this is (2, 2):
a = np.array([[1,2],[3,4]])
<ASSISTANT>{thought} In Python, Numpy provides a method called `shape` which helps to get the dimensions of an array.
{action} PythonInterpreter
{action_input}
```python
import numpy as np
def solution(x):
    # Convert to np.ndarray
    x = np.array(x)
    # Getting the dimensions of the array
    dimensions = x.shape
    return dimensions
assert solution([[1,2],[3,4]]) == (2, 2)
```
<SYSTEM>{response}True
<ASSISTANT> {thought} By running this code, you can get the dimensions of an array.
{finish}
```python
import numpy as np
def solution(x):
    # Convert to np.ndarray
    x = np.array(x)
    # Getting the dimensions of the array
    dimensions = x.shape
    return dimensions
```
"""

ds1000_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='test_column',
    train_split='test',
    test_split='test',
)

ds1000_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{prompt}""",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer, example=ds1000_example),
)

ds1000_eval_cfg = dict(
    evaluator=dict(type=DS1000InterpreterEvaluator),
    pred_role='BOT',
)

# The DS-1000 dataset can be downloaded from
# https://github.com/HKUNLP/DS-1000/blob/main/ds1000_data.zip

# Matplotlib cannot fit this setting
ds1000_datasets = [
    dict(
        abbr=f'ds1000_{lib}',
        type=DS1000Dataset_Interperter,  # bustm share the same format with AFQMC
        path='./data/ds1000_data/',
        libs=f'{lib}',
        reader_cfg=ds1000_reader_cfg,
        infer_cfg=ds1000_infer_cfg,
        eval_cfg=ds1000_eval_cfg,
    )
    for lib in [
        'Pandas',
        'Numpy',
        # 'Tensorflow',  # error using tensorflow, skipped temporarily
        'Scipy',
        'Sklearn',
        'Pytorch',
    ]
]
