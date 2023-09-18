class QwenVLMMBenchPromptConstructor:
    """MMBench prompt constructor for Qwen-VL.

    The output is a dict following the input format of Qwen-VL tokenizer.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, inputs: dict) -> list:
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


class QwenVLChatPromptConstructor:
    """Prompt constructorfor Qwen-VL-Chat."""

    def __init__(self, prompt='') -> None:
        self.prompt = prompt

    def __call__(self, inputs: dict) -> list:
        assert len(inputs['data_samples']) == 1
        format_input = [
            {
                'image': 'This_is_path_to_an_image.'
            },  # Just placeholder for Image Tokens
            {
                'text': self.prompt
            },
        ]
        return format_input


class QwenVLChatVQAPromptConstructor:
    """VQA prompt constructor for Qwen-VL-Chat."""

    def __init__(self, prompt='') -> None:
        self.prompt = prompt

    def __call__(self, inputs: dict) -> list:
        data_samples = inputs['data_samples']
        assert len(data_samples) == 1
        data_sample = data_samples[0]
        question = data_sample.get('question')
        format_input = [
            {
                'image': 'This_is_path_to_an_image.'
            },  # Just placeholder for Image Tokens
            {
                'text': question + self.prompt
            },
        ]
        return format_input


class QwenVLChatScienceQAPromptConstructor:
    """ScienceQA prompt constructor for Qwen-VL-Chat."""
    choice_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}

    def __init__(self, prompt='') -> None:
        self.prompt = prompt

    def __call__(self, inputs: dict) -> list:
        data_samples = inputs['data_samples']
        assert len(data_samples) == 1
        data_sample = data_samples[0]
        question = data_sample.get('question')
        choices = data_sample.get('choices')
        choices = [
            f'({self.choice_mapping[i]}) ' + item
            for i, item in enumerate(choices)
        ]
        choices = 'Choices: ' + ' '.join(choices) + '\n'
        contexts = 'Context: ' + data_sample.get('hint')
        format_input = [
            {
                'image': 'This_is_path_to_an_image.'
            },  # Just placeholder for Image Tokens
            {
                'text': contexts + question + choices + self.prompt
            },
        ]
        return format_input
