import torch


class VisualGLMMMBenchPromptConstructor:
    """MMBench prompt constructor for VisualGLM.

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
            A tuple containing images, prompt, data_samples and image_position.
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


class VisualGLMBasePromptConstructor:
    """Base prompt constructor for VisualGLM.

    The prompt will concat <img> and the given system prompt.
    Args:
        system_prompt (str): System prompt. (Default: '')
    """

    def __init__(self, system_prompt='') -> None:
        self.prompt = system_prompt

    def __call__(self, batch: dict) -> tuple:
        """Construct prompt.

        Args:
            batch (dict): Input data containing image and data_samples.

        Returns:
            A tuple containing images, prompt, data_samples and image_position.
        """

        images = batch.pop('inputs')
        images = torch.stack(images, dim=0)
        data_samples = batch.pop('data_samples')

        # generate text prompt
        img_prompt = '<img></img>'
        prompt = img_prompt + self.prompt
        image_position = prompt.rfind('<img>') + 5

        image_position = 5

        return images, prompt, data_samples, image_position


class VisualGLMVQAPromptConstructor(VisualGLMBasePromptConstructor):
    """VQA prompt constructor for VisualGLM.

    The prompt will concat <img>, the question and the system prompt.
    Args:
        system_prompt (str): System prompt. (Default: '')
    """

    def __init__(self, system_prompt='') -> None:
        super().__init__(system_prompt)

    def __call__(self, batch: dict) -> tuple:
        """Construct prompt.

        Args:
            batch (dict): Input data containing image and data_samples.

        Returns:
            A tuple containing images, prompt, data_samples and image_position.
        """

        images = batch.pop('inputs')
        images = torch.stack(images, dim=0)
        data_samples = batch.pop('data_samples')
        questions = [sample.get('question') for sample in data_samples]

        # generate text prompt
        prompt = [
            '<img></img>Q:{} {}\nA:'.format(question, self.prompt)
            for question in questions
        ]
        image_position = 5

        return images, prompt, data_samples, image_position


class VisualGLMScienceQAPromptConstructor(VisualGLMBasePromptConstructor):
    """ScienceQA prompt constructor for VisualGLM.

    The prompt will concat image and all terms in a question.
    Args:
        system_prompt (str): System prompt. (Default: '')
    """

    choice_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}

    def __init__(self, system_prompt='') -> None:
        super().__init__(system_prompt)

    def __call__(self, batch: dict) -> tuple:
        """Construct prompt.

        Args:
            batch (dict): Input data containing image and data_samples.

        Returns:
            A tuple containing images, prompt, data_samples and image_position.
        """

        images = batch.pop('inputs')
        images = torch.stack(images, dim=0)
        data_samples = batch.pop('data_samples')
        questions = [
            'Q: ' + sample.get('question') + '\n' for sample in data_samples
        ]
        choices = [sample.get('choices') for sample in data_samples]
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

        # generate text prompt
        prompt = [
            '<img></img>' + context + question + choice + self.prompt
            for context, question, choice in zip(contexts, questions, choices)
        ]
        image_position = 5

        return images, prompt, data_samples, image_position


class VisualGLMIconQAPromptConstructor(VisualGLMBasePromptConstructor):
    """IconQA prompt constructor for VisualGLM.

    The prompt will concat <img>, the question and the system prompt.
    Args:
        system_prompt (str): System prompt. (Default: '')
    """

    def __init__(self, system_prompt='') -> None:
        super().__init__(system_prompt)

    def __call__(self, batch: dict) -> tuple:
        """Construct prompt.

        Args:
            batch (dict): Input data containing image and data_samples.

        Returns:
            A tuple containing images, prompt, data_samples and image_position.
        """

        images = batch.pop('inputs')
        images = torch.stack(images, dim=0)
        data_samples = batch.pop('data_samples')
        questions = [
            'Q: ' + sample.get('question') + '\n' for sample in data_samples
        ]
        choices = [sample.get('choices') for sample in data_samples]
        choices = [
            'Options: ' + ', '.join(choice) + '.\n' for choice in choices
        ]  # noqa

        # generate text prompt
        prompt = [
            '<img></img>' + question + choice + self.prompt
            for question, choice in zip(questions, choices)
        ]
        image_position = 5

        return images, prompt, data_samples, image_position
