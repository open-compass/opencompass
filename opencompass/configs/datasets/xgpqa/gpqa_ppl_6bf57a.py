from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.datasets import GPQADataset, GPQAEvaluator
from opencompass.utils import first_option_postprocess

gpqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer')

gpqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            opt: f'Question: {{question}}\n(A){{A}}\n(B){{B}}\n(C){{C}}\n(D){{D}}\nAnswer: {opt}' for opt in ['A', 'B', 'C', 'D']
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

gpqa_eval_cfg = dict(evaluator=dict(type=GPQAEvaluator),
                     pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

gpqa_datasets = []
gpqa_subsets = {
    # 'extended': 'gpqa_extended.csv',
    # 'main': 'gpqa_main.csv',
    'diamond': 'gpqa_diamond.csv'
}

for split in list(gpqa_subsets.keys()):
    gpqa_datasets.append(
        dict(
            abbr='GPQA_' + split,
            type=GPQADataset,
            path='./data/gpqa/',
            name=gpqa_subsets[split],
            reader_cfg=gpqa_reader_cfg,
            infer_cfg=gpqa_infer_cfg,
            eval_cfg=gpqa_eval_cfg)
    )
