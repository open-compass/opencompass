from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator
from opencompass.datasets import IDAM2ICODataset

idam2ico_reader_cfg = dict(
    input_columns=['premise', 'hypothesis'],
    output_column='inference',
    test_split='test')


idam2ico_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Read the given premise and hypithesis, determine whether the "hypothesis" is true (entailment) or false (contradiction) given the "premise". Your answer should be either "entailment" or "contradiction". {premise}\n{hypothesis}'),
                dict(role='BOT', prompt='Inference:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50))

idam2ico_eval_cfg = dict(
    evaluator=dict(type=EMEvaluator),
    pred_role="BOT")

idam2ico_datasets = [
    dict(
        type=IDAM2ICODataset,
        abbr='idam2ico',
        path='./data/id_am2ico/',
        reader_cfg=idam2ico_reader_cfg,
        infer_cfg=idam2ico_infer_cfg,
        eval_cfg=idam2ico_eval_cfg)
]