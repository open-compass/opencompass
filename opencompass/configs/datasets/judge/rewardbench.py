from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import JudgeEvaluator
from opencompass.datasets import RewardBenchDataset


subjective_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='judge',
    )

data_path = './data/judgeeval/rewardbench'
subjective_all_sets = ['llmbar-natural.json', 'llmbar-adver-GPTInst.json', 'hep-go.json', 'refusals-dangerous.json', 'hep-cpp.json', 'mt-bench-easy.json', 'alpacaeval-length.json', 'llmbar-adver-neighbor.json', 'alpacaeval-easy.json', 'hep-java.json', 'llmbar-adver-GPTOut.json', 'mt-bench-hard.json', 'xstest-should-respond.json', 'xstest-should-refuse.json', 'hep-python.json', 'refusals-offensive.json', 'alpacaeval-hard.json', 'llmbar-adver-manual.json', 'hep-js.json', 'math-prm.json', 'hep-rust.json', 'mt-bench-med.json', 'donotanswer.json']
get_rewardbench_datasets = []

for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt='{prompt}'
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=4096),
        )

    rewardbench_eval_cfg = dict(
        evaluator=dict(
            type=JudgeEvaluator,
        ),
    )

    get_rewardbench_datasets.append(
        dict(
            abbr=f'{_name.split(".")[0]}',
            type=RewardBenchDataset,
            path=data_path,
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=rewardbench_eval_cfg,
            mode='singlescore',
        ))
