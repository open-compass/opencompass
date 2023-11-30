from typing import Optional

from mmpretrain.structures import DataSample


class OpenFlamingoMMBenchPromptConstructor:
    """MMBench prompt constructor for OpenFlamingo."""

    def __init__(self) -> None:
        pass

    def __call__(self, data_samples: DataSample) -> tuple:
        """Construct prompt.

        Args:
            data_samples (DataSample): Input data_samples.

        Returns:
            Raw text input (str).
        """
        assert len(data_samples) == 1
        sample = data_samples[0]
        prompts = []
        question = sample.get('question')
        option = sample.get('options')

        prompt = '<image>' + question + ' ' + option + ' ' + 'Answer:'
        if sample.get('context') is not None:
            prompt = sample.get('context') + ' ' + prompt

        prompts.append(prompt)

        return prompts


class OpenFlamingoCaptionPromptConstructor:
    """Caption prompt constructor for OpenFlamingo."""

    def __init__(self, shot_prompt: Optional[str] = None) -> None:
        if shot_prompt:
            self.shot_prompt = shot_prompt
        else:
            self.shot_prompt = (
                'Output:A child holding a flowered umbrella and petting a yak.<|endofchunk|>'  # noqa
                'Output:The child is holding a brush close to his mouth.<|endofchunk|>'  # noqa
            )  # noqa

    def __call__(self, data_samples: DataSample) -> tuple:
        """Construct prompt.

        Args:
            data_samples (DataSample): Input data_samples.

        Returns:
            Raw text input (str).
        """
        assert len(data_samples) == 1
        prompts = []
        prompt = '<image>Output:'
        prompts.append(self.shot_prompt + prompt)
        return prompts


class OpenFlamingoVQAPromptConstructor:
    """VQA prompt constructor for OpenFlamingo."""

    def __init__(self, shot_prompt: Optional[str] = None) -> None:
        if shot_prompt:
            self.shot_prompt = shot_prompt
        else:
            self.shot_prompt = (
                'Question:Is the sky dark? Short Answer:yes<|endofchunk|>'  # noqa: E501
                'Question:What is on the white wall? Short Answer:pipe<|endofchunk|>'  # noqa: E501
            )  # noqa

    def __call__(self, data_samples: DataSample) -> tuple:
        """Construct prompt.

        Args:
            data_samples (DataSample): Input data_samples.

        Returns:
            Raw text input (str).
        """
        prompts = []
        for sample in data_samples:
            question = sample.get('question')
            prompt = '<image>Question:{} Short Answer:'.format(question)
            prompts.append(self.shot_prompt + prompt)
        return prompts


class OpenFlamingoScienceQAPromptConstructor:
    """ScienceQA prompt constructor for OpenFlamingo."""
    choice_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}

    def __init__(self, shot_prompt: Optional[str] = None) -> None:
        if shot_prompt:
            self.shot_prompt = shot_prompt
        else:
            self.shot_prompt = (
                "Context:Question:Which of these states is farthest north? Choices:['(A) West Virginia' '(B) Louisiana' '(C) Arizona' '(D) Oklahoma'] Answer with a single character: A<|endofchunk|>"  # noqa
                'Context:The diagrams below show two pure samples of gas in identical closed, rigid containers. Each colored ball represents one gas particle. Both samples have the same number of particles.'  # noqa
                "Question:Compare the average  kinetic energies of the particles in each sample. Which sample has the higher temperature? Choices:'[(A) neither' '(B) sample A' '(C) sample B'] Answer with a single character: C<|endofchunk|>"  # noqa
            )  # noqa

    def __call__(self, data_samples: DataSample) -> tuple:
        """Construct prompt.

        Args:
            data_samples (DataSample): Input data_samples.

        Returns:
            Raw text input (str).
        """
        assert len(data_samples) == 1
        sample = data_samples[0]
        question = sample.get('question')
        choices = sample.get('choices')
        choices = [
            f'({self.choice_mapping[i]}) ' + item
            for i, item in enumerate(choices)
        ]
        hint = sample.get('hint')
        prompts = []
        prompt = '<image>Context:{} Question:{} Choices:{}'.format(
            hint, question, choices)
        prompt += ' Answer with a single character:'
        prompts.append(self.shot_prompt + prompt)
        return prompts
