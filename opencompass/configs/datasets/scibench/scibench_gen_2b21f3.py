import os
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import ScibenchDataset, scibench_postprocess

scibench_reader_cfg = dict(input_columns=['question'], output_column='answer')

scibench_subsets = [
    'atkins',
    'calculus',
    'chemmc',
    'class',
    'diff',
    'fund',
    'matter',
    'quan',
    'stat',
    'thermo'
]

scibench_datasets = []
for prompt_type in ['zs', 'zs-cot', 'fs', 'fs-cot']:
    for _name in scibench_subsets:
        if prompt_type == 'fs':
            prompt_path = os.path.join(os.path.dirname(__file__), 'lib_prompt', f'{_name}_prompt.txt')
        elif prompt_type == 'fs-cot':
            prompt_path = os.path.join(os.path.dirname(__file__), 'lib_prompt', f'{_name}_sol.txt')
        else:
            prompt_path = None
        if prompt_path is not None:
            with open(prompt_path, 'r') as f:
                _hint = f.read()
        else:
            _hint = ''

        human_prompt = {
            'zs': "Please provide a clear and step-by-step solution for a scientific problem in the categories of Chemistry, Physics, or Mathematics. The problem will specify the unit of measurement, which should not be included in the answer. Express the final answer as a decimal number with three digits after the decimal point. Conclude the answer by stating 'Therefore, the answer is \\boxed[ANSWER].'\n\nProblem: {question}\nAnswer:",
            'zs-cot': "Please provide a clear and step-by-step solution for a scientific problem in the categories of Chemistry, Physics, or Mathematics. The problem will specify the unit of measurement, which should not be included in the answer. Express the final answer as a decimal number with three digits after the decimal point. Conclude the answer by stating 'Therefore, the answer is \\boxed[ANSWER].'\n\nProblem: {question}\nAnswer:Let’s think step by step.",
            'fs': f'{_hint}\n\nProblem 6: {{question}}\nAnswer: ',
            'fs-cot': f'{_hint}\n\nProblem 6: {{question}}\nExplanation for Problem 6: ',
        }[prompt_type]

        scibench_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(role='HUMAN', prompt=human_prompt)
                ])
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=512)
        )

        scibench_eval_cfg = dict(
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(type=scibench_postprocess))


        scibench_datasets.append(
            dict(
                type=ScibenchDataset,
                path='./data/scibench',
                name=_name,
                abbr= f'scibench-{_name}' if prompt_type == 'zs' else f'scibench-{_name}_{prompt_type}',
                reader_cfg=scibench_reader_cfg,
                infer_cfg=scibench_infer_cfg.copy(),
                eval_cfg=scibench_eval_cfg.copy()
            )
        )
