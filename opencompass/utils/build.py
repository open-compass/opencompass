import copy

from mmengine.config import ConfigDict

from opencompass.registry import LOAD_DATASET, MODELS


def build_dataset_from_cfg(dataset_cfg: ConfigDict):
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_cfg.pop('infer_cfg', None)
    dataset_cfg.pop('eval_cfg', None)
    dataset_cfg.pop('abbr', None)
    return LOAD_DATASET.build(dataset_cfg)


def build_model_from_cfg(model_cfg: ConfigDict):
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg.pop('run_cfg', None)
    model_cfg.pop('max_out_len', None)
    model_cfg.pop('batch_size', None)
    model_cfg.pop('abbr', None)
    model_cfg.pop('summarizer_abbr', None)
    model_cfg.pop('pred_postprocessor', None)
    model_cfg.pop('min_out_len', None)
    return MODELS.build(model_cfg)
