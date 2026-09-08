from opencompass.datasets.mrcr import MRCRDataset, MRCREvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

mrcr_1m_subsets = [
    '2needle_in_524288_1048576',
    '4needle_in_524288_1048576',
    '8needle_in_524288_1048576',
]

mrcr_1m_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='gold',
)

mrcr_1m_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': '{context}\n{question}'}
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

mrcr_1m_eval_cfg = dict(
    evaluator=dict(type=MRCREvaluator),
)


mrcr_1m_datasets = []
for subset in mrcr_1m_subsets:

    mrcr_1m_datasets.append(
        dict(
            abbr=f'mrcr_v2_{subset}_1m',
            type=MRCRDataset,
            path='giulio98/MRCR_v2_common',
            subset=subset,
            reader_cfg=mrcr_1m_reader_cfg,
            infer_cfg=mrcr_1m_infer_cfg,
            eval_cfg=mrcr_1m_eval_cfg,
        ))
