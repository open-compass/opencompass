from opencompass.registry import TEXT_POSTPROCESSORS


@TEXT_POSTPROCESSORS.register_module('strategyqa')
def strategyqa_pred_postprocess(text: str) -> str:
    text = text.split('\n\n')[0]
    strategyqa_pre = text.split('So the answer is ')[-1].strip().replace(
        '.', '')
    return strategyqa_pre


@TEXT_POSTPROCESSORS.register_module('strategyqa_dataset')
def strategyqa_dataset_postprocess(text: str) -> str:
    return 'yes' if str(text) == 'True' else 'no'
