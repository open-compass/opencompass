from typing import List

import torch
from mmpretrain.structures import DataSample


class OTTERMMBenchPromptConstructor:
    """Prompt constructor for OTTER on MMBench.

    Args:
        image_prompt (str): Image prompt. Defaults to `''`.
        reply_prompt (str): Reply prompt. Defaults to `''`.
    """

    def __init__(self, user_label: str = '', model_label: str = '') -> None:
        self.image_token = '<image>'
        self.reply_token = '<answer>'
        self.user_label = user_label
        self.model_label = model_label

    def __call__(self, inputs: dict) -> dict:
        """Construct prompt.

        Args:
            inputs (dict): Input data containing image and data_samples.

        Returns:
            dict: A dict containing prompt, images and data_samples.
        """
        images = [image.unsqueeze(0) for image in inputs['inputs']]
        data_samples = [data_sample for data_sample in inputs['data_samples']]
        images = torch.cat(images, dim=0)
        inputs = {'image': images, 'data_samples': data_samples}
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
        data_sample = data_samples[0]
        question = data_sample.get('question')
        options = data_sample.get('options')
        context = data_sample.get('context')
        # e.g. <image>User: What is the color of the sky? A: Blue B: Red C: Green D: Yellow GPT:<answer>  # noqa
        if context is not None:
            prompt = f'{self.image_token}{self.user_label} {context} {question} {options} {self.model_label}:{self.reply_token}'  # noqa
        else:
            prompt = f'{self.image_token}{self.user_label} {question} {options} {self.model_label}:{self.reply_token}'  # noqa

        return prompt


class OTTERCOCOCaotionPromptConstructor(OTTERMMBenchPromptConstructor):
    """Prompt constructor for OTTER on COCO Caption."""

    def _process(self, data_samples: List[DataSample]) -> str:
        # e.g. <image>User: a photo of GPT:<answer>  # noqa
        prompt = f'{self.image_token}{self.user_label} a photo of {self.model_label}:{self.reply_token}'  # noqa
        return prompt


class OTTERScienceQAPromptConstructor(OTTERMMBenchPromptConstructor):
    """Prompt constructor for OTTER on ScienceQA."""

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
        prompt = f'{self.image_token}{self.user_label} {context} {question} {choice} The answer is {self.model_label}:{self.reply_token}'  # noqa
        return prompt


class OTTERVQAPromptConstructor(OTTERMMBenchPromptConstructor):
    """Prompt constructor for OTTER on VQA."""

    def _process(self, data_samples: List[DataSample]) -> str:
        assert len(data_samples) == 1, 'Only support batch size 1.'
        questions = [
            data_sample.get('question') for data_sample in data_samples
        ]
        question = questions[0]
        prompt = f'{self.image_token}{self.user_label} {question}. Answer it with with few words. {self.model_label}:{self.reply_token}'  # noqa
        return prompt


class OTTERVSRPromptConstructor(OTTERMMBenchPromptConstructor):
    """Prompt constructor for OTTER on VSR."""

    def _process(self, data_samples: List[DataSample]) -> str:
        assert len(data_samples) == 1, 'Only support batch size 1.'
        questions = [
            data_sample.get('question') for data_sample in data_samples
        ]
        question = questions[0]
        prompt = f'{self.image_token}{self.user_label} {question}. Is the above description correct? Answer yes or no. {self.model_label}:{self.reply_token}'  # noqa
        return prompt


class OTTERSEEDBenchPromptConstructor(OTTERMMBenchPromptConstructor):

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
        question = questions[0]
        prompt = f'{self.image_token}{self.user_label} {question} {self.model_label}:{self.reply_token}'  # noqa
        return prompt


class OTTERMMEPromptConstructor(OTTERMMBenchPromptConstructor):
    """Prompt constructor for OTTER on MME.

    Args:
        image_prompt (str): Image prompt. Defaults to `''`.
        reply_prompt (str): Reply prompt. Defaults to `''`.
    """

    def _process(self, data_samples: List[DataSample]) -> str:
        """Process data sample to prompt.

        Args:
            data_samples (List[DataSample]): A list of data_samples.

        Returns:
            str: Prompt.
        """
        assert len(data_samples) == 1, 'Only support batch size 1.'
        question = data_samples[0].get('question')
        prompt = f'{self.image_token}{self.user_label} {question} {self.model_label}:{self.reply_token}'  # noqa
        return prompt
