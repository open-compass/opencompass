"""Full EarthSE native dataset configuration.

The public repositories and revisions are frozen by the benchmark evidence.
Judge transport is intentionally injected by the run configuration; setting
``judge_model_cfg`` here would couple the dataset to credentials/endpoints.
"""

from opencompass.datasets import (EarthSEDataset, EarthSEEvaluator,
                                  EarthSEGoldDataset,
                                  EarthSEGoldEvaluator,
                                  EarthSEGoldInferencer)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


SYSTEM_PROMPT = 'You are a helpful assistant.'
SIMILARITY_MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'

qa_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='answer',
)

qa_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(role='system', content=SYSTEM_PROMPT),
            dict(role='user', content='{prompt}'),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


def _qa_dataset(dataset_name, path, revision):
    return dict(
        abbr=dataset_name.lower().replace('-', '_'),
        type=EarthSEDataset,
        path=path,
        revision=revision,
        dataset_name=dataset_name,
        question_types=list(EarthSEDataset.SPLITS),
        prompt_mode='zero-shot',
        reader_cfg=qa_reader_cfg,
        infer_cfg=qa_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(
                type=EarthSEEvaluator,
                judge_model_cfg=None,
                similarity_model_revision=SIMILARITY_MODEL_REVISION,
            ),
        ),
    )


gold_reader_cfg = dict(
    input_columns=['dialogue'],
    output_column='reference',
)

gold_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[dict(expand_column='dialogue')],
        format_variables=False,
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=EarthSEGoldInferencer, multiround=True),
)

earthse_datasets = [
    _qa_dataset(
        'Earth-Iron',
        'ai-earth/Earth-Iron',
        '3b052252a810311b16a6337c9052b41ca8850862',
    ),
    _qa_dataset(
        'Earth-Silver',
        'ai-earth/Earth-Silver',
        'bb2f9cc31911a9d95cca9cedf075fa17ba549b08',
    ),
    dict(
        abbr='earth_gold',
        type=EarthSEGoldDataset,
        path='ai-earth/Earth-Gold',
        revision='caee25d7687582a871a0e09b58a6c8508ce73f2d',
        repetitions=3,
        reader_cfg=gold_reader_cfg,
        infer_cfg=gold_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(
                type=EarthSEGoldEvaluator,
                judge_model_cfg=None,
                similarity_model_revision=SIMILARITY_MODEL_REVISION,
            ),
        ),
    ),
]
