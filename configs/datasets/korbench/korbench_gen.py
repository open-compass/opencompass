from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import korbench_GenInferencer
#from opencompass.datasets import korbenchDataset, korbenchEvaluator
from opencompass.datasets.korbench.cipher import korbenchcipherDataset, korbenchcipherEvaluator
from opencompass.datasets.korbench.counterfactual import korbenchcounterfactualDataset, korbenchcounterfactualEvaluator
from opencompass.datasets.korbench.logic import korbenchlogicDataset, korbenchlogicEvaluator
from opencompass.datasets.korbench.operation import korbenchoperationDataset, korbenchoperationEvaluator
from opencompass.datasets.korbench.puzzle import korbenchpuzzleDataset, korbenchpuzzleEvaluator
import os
# Define task and mode configurations
tasks_with_modes = dict(
    cipher=['zero-shot'],# 'three-shot', 'subquestions'],
    counterfactual=['zero-shot'],# 'three-shot'],
    logic=['zero-shot'],# 'three-shot'],
    operation=['zero-shot'],# 'three-shot'],
    #puzzle=['zero-shot'],# 'three-shot'],
    #mixed=['Multi-Q', 'Multi-R', 'Multi-RQ', 'Multi-RQ Hard']
)
# get the path of this python program
import os
path = os.path.abspath(__file__)

# Base path for the dataset files
base_path = '/home/epsilon/miniforge3/my_opencompass_project/opencompass/data/korbench'


# Define system messages for each task
SYSTEM_MESSAGES = dict(
    default="You are a helpful assistant.",
    cipher="You are an expert in cryptography. Solve the following problem.",
    counterfactual="You are an expert in reasoning about counterfactual scenarios. Answer the following question logically.",
    logic="You are an expert in logical reasoning. Solve the following logic problem.",
    operation="You are an expert in mathematical operations. Perform the required calculations.",
    puzzle="You are an expert in solving puzzles. Solve the following question step by step.",
    mixed="You are an expert in multiple reasoning domains. Answer the following multi-domain question."
)

# Define task-specific prompt templates
PROMPT_TEMPLATES = dict(
    default="{prompt}",
    cipher="### Cipher Problem:\n{prompt}\n### Answer:",
    counterfactual="### Counterfactual Scenario:\n{prompt}\n### Answer:",
    logic="### Logic Problem:\n{prompt}\n### Answer:",
    operation="### Operation Task:\n{prompt}\n### Answer:",
    puzzle="### Puzzle Task:\n{prompt}\n### Answer:",
    mixed="### Multi-Domain Problem:\n{prompt}\n### Answer:"
)

# Factory function to create PromptTemplate
def get_prompt_template(task):
    system_prompt = SYSTEM_MESSAGES.get(task, SYSTEM_MESSAGES['default'])
    task_prompt_template = PROMPT_TEMPLATES.get(task, PROMPT_TEMPLATES['default'])
    # Construct the prompt template
    template = {}
    if system_prompt:
        template['begin'] = [
            dict(
                role='HUMAN',
                prompt=system_prompt
            )
        ]
    template['round'] = [
        dict(
            role='HUMAN',
            prompt=task_prompt_template
        )
    ]
    return dict(
        type=PromptTemplate,
        template=template
    )

# Initialize the list of datasets
korbench_datasets = []

# Iterate over tasks and modes
for task, modes in tasks_with_modes.items():
    for mode in modes:
        # Set the dataset file based on the task and mode
        if task == 'mixed':
            if mode in ['Multi-Q', 'Multi-R', 'Multi-RQ', 'Multi-RQ Hard']:
                data_file = f'{mode}.jsonl'
            else:
                continue  # Skip unsupported modes for 'mixed' task
        else:
            if mode == 'three-shot':
                data_file = 'three-shot.jsonl'
            elif mode == 'subquestions' and task == 'cipher':
                data_file = 'subquestions.json'
            else:
                data_file = 'sample.jsonl'

        # Define the dataset path
        path = os.path.abspath(__file__)
        # Base path for the dataset files
        # base_path = '/home/epsilon/miniforge3/my_opencompass_project/opencompass/data/korbench'
        # use relative path to get the base path
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(path))), 'data/korbench')
        print(f"base_path: {base_path}")
        dataset_path = f'{base_path}/{task}/{data_file}'
        print(f"dataset_path: {dataset_path}")

        # Reader configuration
        reader_cfg = dict(
            input_columns=['prompt'],
            output_column='answer',
        )

        # Inference configuration
        infer_cfg = dict(
            prompt_template=get_prompt_template(task),
            retriever=dict(
                type=ZeroRetriever,
            ),
            inferencer=dict(
                type=korbench_GenInferencer,
                max_out_len=1024,
            )
        )
        
        # Evaluation configuration
        eval_cfg = dict(
            evaluator=dict(
                type=korbenchEvaluator,
            ),
            pred_role='BOT',
        )

        # Assemble the dataset configuration
        dataset = dict(
            type=korbenchDataset,
            abbr=f'korbench_{task}_{mode}',
            path=dataset_path,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg,
        )

        # Append the dataset configuration to the list
        korbench_datasets.append(dataset)


