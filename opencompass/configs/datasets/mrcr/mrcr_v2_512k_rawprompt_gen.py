from opencompass.datasets.mrcr import MRCRDataset, MRCREvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

mrcr_512k_subsets = [
    '2needle_in_262144_524288',
    '4needle_in_262144_524288',
    '8needle_in_262144_524288',
]

mrcr_512k_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='gold',
)

mrcr_512k_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': '{context}\n{question}'}
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

mrcr_512k_eval_cfg = dict(
    evaluator=dict(type=MRCREvaluator),
)


mrcr_512k_datasets = []
for subset in mrcr_512k_subsets:

    mrcr_512k_datasets.append(
        dict(
            abbr=f'mrcr_v2_{subset}_512k',
            type=MRCRDataset,
            path='giulio98/MRCR_v2_common',
            subset=subset,
            reader_cfg=mrcr_512k_reader_cfg,
            infer_cfg=mrcr_512k_infer_cfg,
            eval_cfg=mrcr_512k_eval_cfg,
        ))
