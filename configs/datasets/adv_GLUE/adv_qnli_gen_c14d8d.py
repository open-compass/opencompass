from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import adv_qnliDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess

adv_qnli_reader_cfg = dict(
    input_columns=['question', 'sentence'],
    output_column='label_option',
    train_split='validation',
    test_split='validation')

adv_qnli_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                """Please identify whether the sentence answers the question. The answer should be exactly 'A. yes' or 'B. no'.
question: {question}
sentence: {sentence}
Answer:"""),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

adv_qnli_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_capital_postprocess),
)

adv_qnli_datasets = [
    dict(
        abbr='adv_qnli',
        type=adv_qnliDataset,
        path='adv_glue',
        name='adv_qnli',
        reader_cfg=adv_qnli_reader_cfg,
        infer_cfg=adv_qnli_infer_cfg,
        eval_cfg=adv_qnli_eval_cfg,
    )
]
