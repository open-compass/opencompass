from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import OBQADatasetV2

obqa_reader_cfg = dict(
    input_columns=['question_stem', 'A', 'B', 'C', 'D', 'fact1'],
    output_column='answerKey'
)
obqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            ans: dict(
                round=[
                    dict(
                        role='HUMAN',
                        prompt='We know the fact that {fact1}.\nQuestion: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n'
                    ),
                    dict(role='BOT', prompt=f'Answer: {ans}'),
                ], )
            for ans in ['A', 'B', 'C', 'D']
        }
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)
obqa_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )


obqa_datasets = [
    dict(
        abbr='openbookqa_fact',
        type=OBQADatasetV2,
        path='opencompass/openbookqa_fact',
        name='additional',
        reader_cfg=obqa_reader_cfg,
        infer_cfg=obqa_infer_cfg,
        eval_cfg=obqa_eval_cfg,
    ),
]
