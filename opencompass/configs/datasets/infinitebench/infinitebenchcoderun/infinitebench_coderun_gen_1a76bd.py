from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess
from opencompass.datasets import InfiniteBenchcoderunDataset

InfiniteBench_coderun_reader_cfg = dict(
    input_columns=['context', 'func', 'func_call'],
    output_column='answer',

)

InfiniteBench_coderun_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='You are a helpful assistant.'),
            ],
            round=[
                dict(role='HUMAN', prompt='Following is a set of Python functions. There is a function called named {func}.\n\n{context}\n\nPlease give me the exact number of the return value of {func_call}. Be concise. Your response must end with the final returned value.'),
                dict(role='BOT', prompt=''),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=5)
)

InfiniteBench_coderun_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
    pred_role='BOT'
)

InfiniteBench_coderun_datasets = [
    dict(
        type=InfiniteBenchcoderunDataset,
        abbr='InfiniteBench_coderun',
        path='./data/InfiniteBench/code_run.jsonl',
        reader_cfg=InfiniteBench_coderun_reader_cfg,
        infer_cfg=InfiniteBench_coderun_infer_cfg,
        eval_cfg=InfiniteBench_coderun_eval_cfg)
]
