from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (ELBenchChoiceEvaluator, ELBenchGeneralDataset,
                                  ELBenchMathEvaluator)

# General-capability objective subsets. The prompt text already embeds the
# required answer format (e.g. "ANSWER: [LETTER]" or "\\boxed{}"), so the
# inference prompt is just the question itself.
#   choice subsets -> ELBenchChoiceEvaluator
#   math subsets   -> ELBenchMathEvaluator
elbench_general_subsets = [
    ('mmlu_pro', 'mmlu_pro_sampled', ELBenchChoiceEvaluator),
    ('ceval', 'ceval_sampled', ELBenchChoiceEvaluator),
    ('math_500', 'math_500_sampled', ELBenchMathEvaluator),
    ('aime24', 'aime24_sampled', ELBenchMathEvaluator),
    ('aime25', 'aime25', ELBenchMathEvaluator),
    ('aime26', 'aime26', ELBenchMathEvaluator),
]

elbench_general_reader_cfg = dict(
    input_columns=['question'],
    output_column='target',
)

elbench_general_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='{question}'),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048),
)

elbench_general_datasets = []
for _abbr, _name, _evaluator in elbench_general_subsets:
    elbench_general_datasets.append(
        dict(
            abbr=f'elbench_general_{_abbr}',
            type=ELBenchGeneralDataset,
            path='通用',
            name=_name,
            reader_cfg=elbench_general_reader_cfg,
            infer_cfg=elbench_general_infer_cfg,
            eval_cfg=dict(evaluator=dict(type=_evaluator)),
        ))
