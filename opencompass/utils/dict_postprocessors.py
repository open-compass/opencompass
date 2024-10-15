from opencompass.registry import DICT_POSTPROCESSORS


@DICT_POSTPROCESSORS.register_module('base')
def base_postprocess(output: dict) -> dict:
    return output
