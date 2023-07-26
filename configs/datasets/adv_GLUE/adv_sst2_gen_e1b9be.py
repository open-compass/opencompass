from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import adv_sst2Dataset
from opencompass.utils.text_postprocessors import first_capital_postprocess

adv_sst2_reader_cfg = dict(
    input_columns=['sentence'],
    output_column='label_option',
    train_split='validation',
    test_split='validation')

adv_sst2_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                """For the given sentence, label the sentiment of the sentence as positive or negative. The answer should be exactly 'A. negative' or 'B. positive'.
sentence: {sentence}
Answer:"""),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

adv_sst2_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_capital_postprocess),
)

adv_sst2_datasets = [
    dict(
        abbr='adv_sst2',
        type=adv_sst2Dataset,
        path='adv_glue',
        name='adv_sst2',
        reader_cfg=adv_sst2_reader_cfg,
        infer_cfg=adv_sst2_infer_cfg,
        eval_cfg=adv_sst2_eval_cfg,
    )
]
