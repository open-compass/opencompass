from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import AdvSst2Dataset, AccDropEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess

adv_sst2_reader_cfg = dict(
    input_columns=['sentence'], output_column='label_option')

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
    evaluator=dict(type=AccDropEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

adv_sst2_datasets = [
    dict(
        abbr='adv_sst2',
        type=AdvSst2Dataset,
        path='opencompass/advglue-dev',
        reader_cfg=adv_sst2_reader_cfg,
        infer_cfg=adv_sst2_infer_cfg,
        eval_cfg=adv_sst2_eval_cfg,
    )
]
