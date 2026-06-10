from opencompass.datasets.mrcr import MRCRDataset, MRCREvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

mrcr_subsets = [
    # 2-needle
    '2needle_in_4096_8192',                                                   
    '2needle_in_8192_16384',
    '2needle_in_16384_32768',                                                 
    '2needle_in_32768_65536',                                               
    '2needle_in_65536_131072',
    '2needle_in_131072_262144',
    '2needle_in_262144_524288',
    '2needle_in_524288_1048576',
    # 4-needle
    '4needle_in_4096_8192',
    '4needle_in_8192_16384',
    '4needle_in_16384_32768',
    '4needle_in_32768_65536',
    '4needle_in_65536_131072',
    '4needle_in_131072_262144',
    '4needle_in_262144_524288',
    '4needle_in_524288_1048576',
    # 8-needle
    '8needle_in_4096_8192',
    '8needle_in_8192_16384',
    '8needle_in_16384_32768',
    '8needle_in_32768_65536',
    '8needle_in_65536_131072',
    '8needle_in_131072_262144',
    '8needle_in_262144_524288',
    '8needle_in_524288_1048576',
]

mrcr_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='gold',
)

mrcr_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': '{context}\n{question}'}
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

mrcr_eval_cfg = dict(
    evaluator=dict(type=MRCREvaluator),
)


mrcr_datasets = []
for subset in mrcr_subsets:

    mrcr_datasets.append(
        dict(
            abbr=f'mrcr_v2_{subset}',
            type=MRCRDataset,
            path='giulio98/MRCR_v2_common',
            subset=subset,
            reader_cfg=mrcr_reader_cfg,
            infer_cfg=mrcr_infer_cfg,
            eval_cfg=mrcr_eval_cfg,
        ))
