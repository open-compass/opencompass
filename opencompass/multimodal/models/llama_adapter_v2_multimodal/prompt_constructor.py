from typing import List

from mmpretrain.structures import DataSample


class LlamaAadapterMMBenchPromptConstructor:
    """Prompt constructor for Llama Adapter v2 on MMBench.

    Args:
        image_prompt (str): Image prompt. Defaults to `''`.
        reply_prompt (str): Reply prompt. Defaults to `''`.
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
        # import pdb;pdb.set_trace()
        question = [
            data_sample.get('question') for data_sample in data_samples
        ]
        options = [data_sample.get('options') for data_sample in data_samples]
        if data_samples[0].get('context') is not None:
            context = [
                data_sample.get('context') for data_sample in data_samples
            ]
        else:
            context = [''] * len(data_samples)
        prompts = []
        for cur_context, cur_question, cur_options in zip(
                context, question, options):
            prompts.append(cur_context + ' ' + cur_question + ' ' +
                           cur_options)  # noqa
        return prompts
