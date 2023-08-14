from typing import List
import torch

from opencompass.registry import MM_MODELS
@MM_MODELS.register_module('visualglm-promptconstructor')
class VisualGLMPromptConstructor:
    """Prompt constructor for VisualGLM.
    The overall prompt will be formulated as 
    "system_prompt"+"human_prompt"+"image_prompt"+question+"assistant+prompt".
    Args:
        system_prompt (str): System prompt.
        human_prompt (str): Human prompt.
        image_prompt (str): Image prompt.
        assistant_prompt (str): Assistant prompt.
    """

    def __init__(self, system_prompt: str = '', human_prompt:str = 'Q:', image_prompt: str = '<img></img>', assistant_prompt: str = 'A:') -> None:
        self.image_prompt = image_prompt
        self.system_prompt = system_prompt
        self.human_prompt = human_prompt
        self.assistant_prompt = assistant_prompt

    def __call__(self, batch: dict) -> tuple:
        """Construct prompt.

        Args:
            batch (dict): Input data containing image and data_samples.

        Returns:
            tuple: A tuple containing prompt, images and data_samples.
        """

        images = batch.pop('inputs')
        images = torch.stack(images, dim=0)

        data_samples = batch.pop('data_samples')
        questions = [sample.get('question') for sample in data_samples]
        options = [sample.get('options') for sample in data_samples]

        # generate text prompt
        prompt = [
            '{}{}{}{}{}{}'.format(self.system_prompt, self.image_prompt, self.human_prompt, question, option, self.assistant_prompt)
            for question, option in zip(questions, options)
        ]
        
        image_position = 5

        return images, prompt, data_samples, image_position
