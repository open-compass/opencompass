from opencompass.datasets.mrcr import MRCRDataset, MRCREvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

mrcr_16k_subsets = [
    '2needle_in_8192_16384',
    '4needle_in_8192_16384',
    '8needle_in_8192_16384',
]

mrcr_16k_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='gold',
)

mrcr_16k_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': '{context}\n{question}'}
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

mrcr_16k_eval_cfg = dict(
    evaluator=dict(type=MRCREvaluator),
)


mrcr_16k_datasets = []
for subset in mrcr_16k_subsets:

    mrcr_16k_datasets.append(
        dict(
            abbr=f'mrcr_v2_{subset}',
            type=MRCRDataset,
            path='giulio98/MRCR_v2_common',
            subset=subset,
            reader_cfg=mrcr_16k_reader_cfg,
            infer_cfg=mrcr_16k_infer_cfg,
            eval_cfg=mrcr_16k_eval_cfg,
        ))
