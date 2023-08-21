import torch


class VisualGLMPromptConstructor:
    """Prompt constructor for VisualGLM.

    The overall prompt will be formulated as
    "system_prompt"+"human_prompt"+"image_prompt"+question+"assistant+prompt".
    Args:
        system_prompt (str): System prompt. (Default: '')
        human_prompt (str): Human prompt. (Default: 'Q:')
        image_prompt (str): Image prompt. (Default: '<img></img>')
        assistant_prompt (str): Assistant prompt. (Default: 'A:')
    """

    def __init__(self,
                 system_prompt: str = '',
                 human_prompt: str = 'Q:',
                 image_prompt: str = '<img></img>',
                 assistant_prompt: str = 'A:') -> None:
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
        contexts = [sample.get('context') for sample in data_samples]
        contexts = [c if c else '' for c in contexts]

        # generate text prompt
        prompt = [
            '{}{}{}{}{}{}{}'.format(self.system_prompt, self.image_prompt,
                                    self.human_prompt, context, question,
                                    option, self.assistant_prompt)
            for context, question, option in zip(contexts, questions, options)
        ]

        image_position = 5

        return images, prompt, data_samples, image_position
