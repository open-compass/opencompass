from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import ProcessBenchEvaluator, ProcessBenchEvalDataset

PROCESSBENCH_CRITIQUE_PROMPT = """
The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \\boxed{{}}.
"""



processbench_reader_cfg = dict(input_columns=['problem, tagged_response, label'], output_column='label', test_split='test')

processbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt=PROCESSBENCH_CRITIQUE_PROMPT),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

processbench_eval_cfg = dict(
    evaluator=dict(
        type=ProcessBenchEvaluator,
    )
)

subsets = ['gsm8k', 'math'] #, , 'olympiadbench', 'omnimath']

processbench_datasets = []

for subset in subsets:
    processbench_datasets.append(
        dict(
            type=ProcessBenchEvalDataset,
            abbr=f'processbench_{subset}',
            path='Qwen/ProcessBench',
            subset=subset,
            reader_cfg=processbench_reader_cfg,
            infer_cfg=processbench_infer_cfg,
            eval_cfg=processbench_eval_cfg)
    )
