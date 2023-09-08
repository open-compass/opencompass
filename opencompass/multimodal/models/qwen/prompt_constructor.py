class QwenVLMMBenchPromptConstructor:
    """MMBench prompt constructor for Qwen-VL.

    The output is a dict following the input format of Qwen-VL tokenizer.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, inputs: dict) -> str:
        data_samples = inputs['data_samples']
        assert len(data_samples) == 1
        data_sample = data_samples[0]
        question = data_sample.get('question')
        options = data_sample.get('options')
        context = data_sample.get('context')
        if context is not None:
            prompt = context + ' ' + question + ' ' + options
        else:
            prompt = question + ' ' + options
        format_input = [
            {
                'image': 'This_is_path_to_an_image.'
            },  # Just placeholder for Image Tokens
            {
                'text': prompt
            },
        ]
        return format_input
