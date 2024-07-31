from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import AdvMnliMMDataset, AccDropEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess

adv_mnli_mm_reader_cfg = dict(
    input_columns=['premise', 'hypothesis'], output_column='label_option')

adv_mnli_mm_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                """Please identify whether the premise entails the hypothesis. The answer should be exactly 'A. yes', 'B. maybe' or 'C. no'.
premise: {premise}
hypothesis: {hypothesis}
Answer:"""),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

adv_mnli_mm_eval_cfg = dict(
    evaluator=dict(type=AccDropEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABC'),
)

adv_mnli_mm_datasets = [
    dict(
        abbr='adv_mnli_mm',
        type=AdvMnliMMDataset,
        path='opencompass/advglue-dev',
        reader_cfg=adv_mnli_mm_reader_cfg,
        infer_cfg=adv_mnli_mm_infer_cfg,
        eval_cfg=adv_mnli_mm_eval_cfg,
    )
]
