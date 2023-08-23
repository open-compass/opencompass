from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import Registry

PARTITIONERS = Registry('partitioner', locations=['opencompass.partitioners'])
RUNNERS = Registry('runner', locations=['opencompass.runners'])
TASKS = Registry('task', locations=['opencompass.tasks'])
MODELS = Registry('model', locations=['opencompass.models'])
# TODO: LOAD_DATASET -> DATASETS
LOAD_DATASET = Registry('load_dataset', locations=['opencompass.datasets'])
TEXT_POSTPROCESSORS = Registry(
    'text_postprocessors', locations=['opencompass.utils.text_postprocessors'])
EVALUATORS = Registry('evaluators', locations=['opencompass.evaluators'])

ICL_INFERENCERS = Registry('icl_inferencers',
                           locations=['opencompass.openicl.icl_inferencer'])
ICL_RETRIEVERS = Registry('icl_retrievers',
                          locations=['opencompass.openicl.icl_retriever'])
ICL_DATASET_READERS = Registry(
    'icl_dataset_readers',
    locations=['opencompass.openicl.icl_dataset_reader'])
ICL_PROMPT_TEMPLATES = Registry(
    'icl_prompt_templates',
    locations=['opencompass.openicl.icl_prompt_template'])
ICL_EVALUATORS = Registry('icl_evaluators',
                          locations=['opencompass.openicl.icl_evaluator'])
DATASETS = Registry('mm_datasets',
                    parent=MMENGINE_DATASETS,
                    locations=['opencompass.multimodal.datasets'])
METRICS = Registry('metric',
                   parent=MMENGINE_METRICS,
                   locations=['opencompass.metrics'])
MM_MODELS = Registry('mm_model',
                     parent=MMENGINE_MODELS,
                     locations=['opencompass.multimodal.models'])
TOT_WRAPPER = Registry('tot_wrapper', locations=['opencompass.datasets'])
