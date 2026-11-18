from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import RPCBenchDataset
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.evaluator import GenericLLMEvaluator

# RPC-Bench text-only track, Claim_Verification (525 True/False questions on
# test). The inference prompt is copied verbatim from the official llm.py
# (placeholder {content} renamed to {paper_content}, see RPCBenchDataset).
# The official eval.py scores this category by exact string match; here an
# LLM judge decides whether the final verdict matches the gold label instead,
# so extra reasoning or formatting around True/False is not dropped.

rpcbench_cv_reader_cfg = dict(
    input_columns=['question', 'paper_content'],
    output_column='answer',
)

RPCBENCH_CV_INFER_TEMPLATE = """You are an academic judgment specialist assigned to classify the following statement as strictly 'True' or 'False' based exclusively on the content of the provided research paper. Carefully read and analyze the entire paper. Use only evidence directly from the text—do not incorporate external knowledge, assumptions, or subjective reasoning.

Output Requirements:
- Respond SOLELY with 'True' or 'False'
- No explanations, disclaimers, or supplementary text

Statement:
{question}

Paper:
{paper_content}"""

rpcbench_cv_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': RPCBENCH_CV_INFER_TEMPLATE},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

RPCBENCH_CV_JUDGE_TEMPLATE = """You are an expert academic answer evaluator. A model was asked to classify a statement about a research paper as strictly 'True' or 'False' based on the paper. You will be given the title and abstract of the paper, the statement, the gold-standard label, and the predicted answer.

Determine the final verdict (True or False) expressed by the predicted answer and check whether it matches the gold-standard label.

Evaluation criteria:
1. The gold-standard label is always correct. Do not verify the statement yourself.
2. Ignore reasoning, explanations, formatting, quotes, punctuation and letter case; only the final verdict matters.
3. Answers such as 'Yes'/'Correct' or 'No'/'Incorrect' count as True or False respectively when the intended verdict is unambiguous.
4. If the predicted answer gives no verdict, gives both verdicts, refuses to answer, or is truncated before a verdict, it is INCORRECT.

Grade the predicted answer as one of:
A: CORRECT
B: INCORRECT
Just return the letter "A" or "B", with no text around it.

<paper-title>{title}</paper-title>

<paper-abstract>{abstract}</paper-abstract>

<statement>{question}</statement>

<gold-label>{answer}</gold-label>

<predicted-answer>{prediction}</predicted-answer>
"""

rpcbench_cv_eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                {'role': 'user', 'content': RPCBENCH_CV_JUDGE_TEMPLATE},
            ],
        ),
        dataset_cfg=dict(
            type=RPCBenchDataset,
            path='zai-org/RPC-Bench',
            split='test',
            subset='cv',
            reader_cfg=rpcbench_cv_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
)

rpcbench_cv_datasets = [
    dict(
        type=RPCBenchDataset,
        abbr='rpcbench_test_cv',
        path='zai-org/RPC-Bench',
        split='test',
        subset='cv',
        reader_cfg=rpcbench_cv_reader_cfg,
        infer_cfg=rpcbench_cv_infer_cfg,
        eval_cfg=rpcbench_cv_eval_cfg,
    )
]
