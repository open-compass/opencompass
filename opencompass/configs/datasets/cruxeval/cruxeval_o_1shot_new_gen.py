from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CRUXEvalDataset, CRUXEvalOEvaluator, cruxeval_o_postprocess

# Reader config: the dataset has 800 samples with fields code/input/output/id.
cruxeval_o_reader_cfg = dict(
    input_columns=['code', 'input'], output_column='output')

# 1-shot "direct output (Phind)" prompt for base models, adapted from
# facebookresearch/cruxeval `prompts.make_direct_output_prompt_phind` by
# keeping a single in-context example.
# The prompt ends with `assert f({input}) ==` (no trailing space) so the
# base model directly continues with the output value, followed by `# done`.
cruxeval_o_prompt = '''Based on the given Python code, which may contain errors, complete the assert statement with the output when executing the code on the given test case. Do NOT output any extra information, even if the function is incorrect or incomplete. Output "# done" after the assertion.

def f(n):
    return n
assert f(17) == 17 # done

{code}
assert f({input}) =='''

cruxeval_o_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template=cruxeval_o_prompt),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, stopping_criteria=['# done']),
)

cruxeval_o_eval_cfg = dict(
    evaluator=dict(type=CRUXEvalOEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=cruxeval_o_postprocess),
)

cruxeval_o_datasets = [
    dict(
        type=CRUXEvalDataset,
        abbr='cruxeval_o',
        path='cruxeval-org/cruxeval',
        reader_cfg=cruxeval_o_reader_cfg,
        infer_cfg=cruxeval_o_infer_cfg,
        eval_cfg=cruxeval_o_eval_cfg,
    )
]
