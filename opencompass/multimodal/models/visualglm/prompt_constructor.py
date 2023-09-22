class VisualGLMMMBenchPromptConstructor:
    """MMBench prompt constructor for VisualGLM.

    Args:
        system_prompt (str): System prompt. (Default: '')
        human_prompt (str): Human prompt. (Default: 'Q:')
        assistant_prompt (str): Assistant prompt. (Default: 'A:')
    """

    def __init__(self,
                 system_prompt: str = '',
                 human_prompt: str = 'Q:',
                 assistant_prompt: str = 'A:') -> None:
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

        assert len(batch['inputs']) == 1
        image = batch.pop('inputs')[0].unsqueeze(0)
        data_sample = batch.pop('data_samples')[0]
        img_prompt = '<img></img>'
        if data_sample.get('context') is not None:
            prompt = img_prompt + self.system_prompt + self.human_prompt + data_sample.context + ' ' + data_sample.question + ' ' + data_sample.options  # noqa
        else:
            prompt = img_prompt + self.system_prompt + self.human_prompt + data_sample.question + ' ' + data_sample.options  # noqa
        prompt += self.assistant_prompt
        image_position = prompt.rfind('<img>') + 5

        return image, prompt, data_sample, image_position


class VisualGLMBasePromptConstructor:
    """Base prompt constructor for VisualGLM.

    The prompt will concat <img> and the given system prompt.
    Args:
        system_prompt (str): System prompt. (Default: '')
        human_prompt (str): Human prompt. (Default: 'Q:')
        assistant_prompt (str): Assistant prompt. (Default: 'A:')
    """

    def __init__(self,
                 system_prompt: str = '',
                 human_prompt: str = 'Q:',
                 assistant_prompt: str = 'A:') -> None:
        self.prompt = system_prompt
        self.human_prompt = human_prompt
        self.assistant_prompt = assistant_prompt

    def __call__(self, batch: dict) -> tuple:
        """Construct prompt.

        Args:
            batch (dict): Input data containing image and data_samples.

        Returns:
            A tuple containing images, prompt, data_samples and image_position.
        """

        assert len(batch['inputs']) == 1
        image = batch.pop('inputs')[0].unsqueeze(0)
        data_sample = batch.pop('data_samples')[0]

        # generate text prompt
        prompt = '<img></img>' + self.human_prompt + self.prompt + self.assistant_prompt  # noqa

        image_position = prompt.rfind('<img>') + 5

        return image, prompt, data_sample, image_position


class VisualGLMVQAPromptConstructor(VisualGLMBasePromptConstructor):
    """VQA prompt constructor for VisualGLM.

    The prompt will concat <img>, the question and the system prompt.
    Args:
        system_prompt (str): System prompt. (Default: '')
        human_prompt (str): Human prompt. (Default: 'Q:')
        assistant_prompt (str): Assistant prompt. (Default: 'A:')
    """

    def __init__(self,
                 system_prompt='',
                 human_prompt: str = 'Q:',
                 assistant_prompt: str = 'A:') -> None:
        super().__init__(system_prompt, human_prompt, assistant_prompt)

    def __call__(self, batch: dict) -> tuple:
        """Construct prompt.

        Args:
            batch (dict): Input data containing image and data_samples.

        Returns:
            A tuple containing images, prompt, data_samples and image_position.
        """

        assert len(batch['inputs']) == 1
        image = batch.pop('inputs')[0].unsqueeze(0)
        data_sample = batch.pop('data_samples')[0]

        # generate text prompt
        question = data_sample.get('question')
        prompt = '<img></img>' + self.human_prompt + question + self.prompt
        prompt += '\n' + self.assistant_prompt

        image_position = prompt.rfind('<img>') + 5

        return image, prompt, data_sample, image_position


class VisualGLMScienceQAPromptConstructor(VisualGLMBasePromptConstructor):
    """ScienceQA prompt constructor for VisualGLM.

    The prompt will concat image and all terms in a question.
    Args:
        system_prompt (str): System prompt. (Default: '')
        human_prompt (str): Human prompt. (Default: 'Q:')
        assistant_prompt (str): Assistant prompt. (Default: 'A:')
    """

    choice_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}

    def __init__(self,
                 system_prompt='',
                 human_prompt: str = 'Q:',
                 assistant_prompt: str = 'A:') -> None:
        super().__init__(system_prompt, human_prompt, assistant_prompt)

    def __call__(self, batch: dict) -> tuple:
        """Construct prompt.

        Args:
            batch (dict): Input data containing image and data_samples.

        Returns:
            A tuple containing images, prompt, data_samples and image_position.
        """

        assert len(batch['inputs']) == 1
        image = batch.pop('inputs')[0].unsqueeze(0)
        data_sample = batch.pop('data_samples')[0]

        questions = 'Question: ' + data_sample.get('question')
        choices = data_sample.get('choices')
        choices = [
            f'({self.choice_mapping[i]}) ' + item
            for i, item in enumerate(choices)
        ]
        choices = 'Choices: ' + ' '.join(choices) + '\n'
        contexts = 'Context: ' + data_sample.get('hint') + '\n'

        # generate text prompt
        prompt = '<img></img>' + self.human_prompt + contexts + questions + choices + self.prompt + self.assistant_prompt  # noqa
        image_position = prompt.rfind('<img>') + 5

        return image, prompt, data_sample, image_position


class VisualGLMIconQAPromptConstructor(VisualGLMBasePromptConstructor):
    """IconQA prompt constructor for VisualGLM.

    The prompt will concat <img>, the question and the system prompt.
    Args:
        system_prompt (str): System prompt. (Default: '')
        human_prompt (str): Human prompt. (Default: 'Q:')
        assistant_prompt (str): Assistant prompt. (Default: 'A:')
    """

    def __init__(self,
                 system_prompt='',
                 human_prompt: str = 'Q:',
                 assistant_prompt: str = 'A:') -> None:
        super().__init__(system_prompt, human_prompt, assistant_prompt)

    def __call__(self, batch: dict) -> tuple:
        """Construct prompt.

        Args:
            batch (dict): Input data containing image and data_samples.

        Returns:
            A tuple containing images, prompt, data_samples and image_position.
        """

        assert len(batch['inputs']) == 1
        image = batch.pop('inputs')[0].unsqueeze(0)
        data_sample = batch.pop('data_samples')[0]

        questions = data_sample.get('question') + '\n'
        choices = data_sample.get('choices')
        choices = 'Options: ' + ', '.join(choices) + '.\n'

        # generate text prompt
        prompt = '<img></img>' + self.human_prompt + questions + choices + self.prompt + self.assistant_prompt  # noqa
        image_position = prompt.rfind('<img>') + 5

        return image, prompt, data_sample, image_position
