from opencompass.datasets.mrcr import MRCRDataset, MRCREvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

mrcr_256k_subsets = [
    '2needle_in_131072_262144',
    '4needle_in_131072_262144',
    '8needle_in_131072_262144',
]

mrcr_256k_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='gold',
)

mrcr_256k_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': '{context}\n{question}'}
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

mrcr_256k_eval_cfg = dict(
    evaluator=dict(type=MRCREvaluator),
)


mrcr_256k_datasets = []
for subset in mrcr_256k_subsets:

    mrcr_256k_datasets.append(
        dict(
            abbr=f'mrcr_v2_{subset}_256k',
            type=MRCRDataset,
            path='giulio98/MRCR_v2_common',
            subset=subset,
            reader_cfg=mrcr_256k_reader_cfg,
            infer_cfg=mrcr_256k_infer_cfg,
            eval_cfg=mrcr_256k_eval_cfg,
        ))
