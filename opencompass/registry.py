from typing import Callable, List, Optional, Type, Union

from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import Registry as OriginalRegistry


class Registry(OriginalRegistry):

    # override the default force behavior
    def register_module(
            self,
            name: Optional[Union[str, List[str]]] = None,
            force: bool = True,
            module: Optional[Type] = None) -> Union[type, Callable]:
        return super().register_module(name, force, module)


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
METRICS = Registry('metric',
                   parent=MMENGINE_METRICS,
                   locations=['opencompass.metrics'])
TOT_WRAPPER = Registry('tot_wrapper', locations=['opencompass.datasets'])


def build_from_cfg(cfg):
    """A helper function that builds object with MMEngine's new config."""
    return PARTITIONERS.build(cfg)
