from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CoderEvalDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess


CoderEval_reader_cfg = dict(
    input_columns="input",
    output_column=None,
)

CoderEval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt="Please help me complete the following function.\n**Note: only return the function to me, no other description.**\n```python\n{input}\n```"),
                dict(role="BOT", prompt="{answer}"),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

CoderEval_eval_cfg = dict(evaluator=dict(type=AccEvaluator),
                        pred_role="BOT",
                        pred_postprocessor=dict(type=first_capital_postprocess))

files = ['CEPythonHumanLabel', 'CEPythonRaw']
CoderEval_datasets = []

for _file in files:
    CoderEval_datasets.append(
        dict(
            type=CoderEvalDataset,
            abbr=_file,
            test_path=f"data/CoderEval/{_file}.jsonl",
            reader_cfg=CoderEval_reader_cfg,
            infer_cfg=CoderEval_infer_cfg,
            eval_cfg=CoderEval_eval_cfg,
        )
    )

del _file