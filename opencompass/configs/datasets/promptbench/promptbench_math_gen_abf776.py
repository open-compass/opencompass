from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AttackInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')

original_prompt_list = [
        'Solve the following math question about',
        'Determine the solution to this mathematical problem related to',
        'Calculate the answer to the following math query about',
        'Find the solution for this mathematical challenge with',
        'Compute the result of this math task concerning',
        'Resolve the following mathematical question associated with',
        'Work out the answer to this math problem featuring',
        'Figure out the solution for the following mathematical task with',
        'Obtain the result for this math question regarding',
        'Evaluate the following mathematical problem that includes',
]

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='{adv_prompt} {problem}:'),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AttackInferencer, original_prompt_list=original_prompt_list,max_out_len=512, adv_key='adv_prompt'))

math_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator), pred_postprocessor=dict(type=math_postprocess))

math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math',
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg)
]
