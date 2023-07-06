from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import siqaDataset_V2
from opencompass.utils.text_postprocessors import first_capital_postprocess

siqa_reader_cfg = dict(
    input_columns=["context", "question", "answerA", "answerB", "answerC"],
    output_column="label",
    test_split="validation")

siqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    "{context}\nQuestion: {question}\nA. {answerA}\nB. {answerB}\nC. {answerC}\nAnswer:"
                )
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

siqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_capital_postprocess),
)

siqa_datasets = [
    dict(
        abbr="siqa",
        type=siqaDataset_V2,
        path="social_i_qa",
        reader_cfg=siqa_reader_cfg,
        infer_cfg=siqa_infer_cfg,
        eval_cfg=siqa_eval_cfg)
]
