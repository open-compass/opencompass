from typing import List

from mmpretrain.structures import DataSample


class InstructBlipMMBenchPromptConstructor:
    """Prompt constructor for InstructBlip on MMBench.

    Args:
        image_prompt (str): Image prompt.
        reply_prompt (str): Reply prompt.
    """

    def __init__(self, image_prompt: str = '', reply_prompt: str = '') -> None:
        self.image_prompt = image_prompt
        self.reply_prompt = reply_prompt

    def __call__(self, inputs: dict) -> dict:
        """Construct prompt.

        Args:
            inputs (dict): Input data containing image and data_samples.

        Returns:
            dict: A dict containing prompt, images and data_samples.
        """
        data_samples = inputs['data_samples']
        prompt = self._process(data_samples)
        inputs.update({'prompt': prompt})

        return inputs

    def _process(self, data_samples: List[DataSample]) -> str:
        """Process data sample to prompt.

        Args:
            data_samples (List[DataSample]): A list of data_samples.

        Returns:
            str: Prompt.
        """
        assert len(data_samples) == 1, 'Only support batch size 1.'
        questions = [
            data_sample.get('question') for data_sample in data_samples
        ]
        options = [data_sample.get('options') for data_sample in data_samples]
        contexts = [data_sample.get('context') for data_sample in data_samples]
        question = questions[0]
        option = options[0]
        context = contexts[0]
        if context is not None:
            prompt = self.image_prompt + ' ' + context + ' ' + question + ' ' + option + ' ' + self.reply_prompt  # noqa
        else:
            prompt = self.image_prompt + ' ' + question + ' ' + option + ' ' + self.reply_prompt  # noqa
        return prompt


class InstructBlipCOCOCaotionPromptConstructor(
        InstructBlipMMBenchPromptConstructor):
    """Prompt constructor for InstructBlip on COCO Caption."""

    def _process(self, data_samples: List[DataSample]) -> str:
        assert len(data_samples) == 1, 'Only support batch size 1.'
        prompt = self.image_prompt + ' ' + 'a photo of' + self.reply_prompt
        return prompt


class InstructBlipVQAPromptConstructor(InstructBlipMMBenchPromptConstructor):
    """Prompt constructor for InstructBlip on VQA."""

    def _process(self, data_samples: List[DataSample]) -> str:
        assert len(data_samples) == 1, 'Only support batch size 1.'
        questions = [
            data_sample.get('question') for data_sample in data_samples
        ]
        question = questions[0]
        prompt = self.image_prompt + ' ' + question + ' ' + 'Answer this question in a single word.' + ' ' + self.reply_prompt  # noqa
        return prompt


class InstructBlipScienceQAPromptConstructor(
        InstructBlipMMBenchPromptConstructor):
    """Prompt constructor for InstructBlip on ScienceQA."""

    choice_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}

    def _process(self, data_samples: List[DataSample]) -> str:
        assert len(data_samples) == 1, 'Only support batch size 1.'
        questions = [
            'Question: ' + data_sample.get('question') + '\n'
            for data_sample in data_samples
        ]  # noqa
        choices = [data_sample.get('choices') for data_sample in data_samples]
        choices = [[
            f'({self.choice_mapping[i]}) ' + item
            for i, item in enumerate(choice)
        ] for choice in choices]
        choices = [
            'Choices: ' + ' '.join(choice) + '\n' for choice in choices
        ]  # noqa
        contexts = [
            'Context: ' + data_sample.get('hint') + '\n'
            for data_sample in data_samples
        ]  # noqa
        question = questions[0]
        choice = choices[0]
        context = contexts[0]
        prompt = self.image_prompt + ' ' + context + ' ' + question + ' ' + choice + self.reply_prompt + ' ' + 'The answer is'  # noqa
        return prompt


class InstructBlipVSRPromptConstructor(InstructBlipMMBenchPromptConstructor):
    """Prompt constructor for InstructBlip on VSR."""

    def _process(self, data_samples: List[DataSample]) -> str:
        assert len(data_samples) == 1, 'Only support batch size 1.'
        questions = [
            data_sample.get('question') for data_sample in data_samples
        ]
        question = questions[0]
        prompt = self.image_prompt + ' ' + question + ' ' + 'Is the above description correct? Answer yes or no.' + ' ' + self.reply_prompt  # noqa
        return prompt
