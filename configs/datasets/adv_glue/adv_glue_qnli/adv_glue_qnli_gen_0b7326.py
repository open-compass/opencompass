from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import AdvQnliDataset, AccDropEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess

adv_qnli_reader_cfg = dict(
    input_columns=['question', 'sentence'], output_column='label_option')

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
    evaluator=dict(type=AccDropEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

adv_qnli_datasets = [
    dict(
        abbr='adv_qnli',
        type=AdvQnliDataset,
        path='opencompass/advglue-dev',
        reader_cfg=adv_qnli_reader_cfg,
        infer_cfg=adv_qnli_infer_cfg,
        eval_cfg=adv_qnli_eval_cfg,
    )
]
