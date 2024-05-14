from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import InfiniteBenchendiaDataset, InfiniteBenchendiaEvaluator

InfiniteBench_endia_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='answer',

)

InfiniteBench_endia_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='You are a helpful assistant.'),
            ],
            round=[
                dict(role='HUMAN', prompt='Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\nThe dialogue:\n\n---\n\n{context}\n\n---\n\nEnd of dialogue.\n\nWhich character is most likely \"$$MASK$$\"? Just say the name used by the scriptwriter (before the colon marks) of one single character and nothing else.'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=40)
)

InfiniteBench_endia_eval_cfg = dict(
    evaluator=dict(type=InfiniteBenchendiaEvaluator),
    pred_role='BOT'
)

InfiniteBench_endia_datasets = [
    dict(
        type=InfiniteBenchendiaDataset,
        abbr='InfiniteBench_endia',
        path='./data/InfiniteBench/longdialogue_qa_eng.jsonl',
        reader_cfg=InfiniteBench_endia_reader_cfg,
        infer_cfg=InfiniteBench_endia_infer_cfg,
        eval_cfg=InfiniteBench_endia_eval_cfg)
]
