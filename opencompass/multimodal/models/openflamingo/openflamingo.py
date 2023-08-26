from typing import List, Optional, Union

import mmengine
import torch
from mmpretrain.models.multimodal import Flamingo
from mmpretrain.structures import DataSample

from opencompass.registry import MM_MODELS


@MM_MODELS.register_module('openflamingo')
class OpenFlamingoInferencer(Flamingo):
    """Inference code of OpenFlamingo.

    Args:
        prompt_constructor (optional, dict): The config of prompt constructor.
            Defaults to None.
        post_processor (optional, dict): The config of post processor.
            Defaults to None.
        mode (str): The mode of inference. Defaults to 'generation'.
    """

    def __init__(self,
                 prompt_constructor: Optional[dict] = None,
                 post_processor: Optional[dict] = None,
                 mode: str = 'generation',
                 **kwargs):
        super().__init__(**kwargs)
        if prompt_constructor is not None:
            self.prompt_constructor = mmengine.registry.build_from_cfg(
                prompt_constructor, MM_MODELS)
        if post_processor is not None:
            self.post_processor = mmengine.registry.build_from_cfg(
                post_processor, MM_MODELS)
        self.mode = mode

    def preprocess_text(self, data_samples: List[DataSample],
                        device: torch.device) -> List[DataSample]:
        """Preprocess text in advance before fed into language model.

        Args:
            data_samples (List[DataSample]): The annotation
                data of every samples. Defaults to None.
            device (torch.device): Device for text to put on.

        Returns:
            List[DataSample]: Return list of data samples.
        """
        prompts = []
        for sample in data_samples:
            question = sample.get('question')
            option = sample.get('options')

            prompt = '<image>' + question + ' ' + option + ' ' + 'Answer:'
            if data_samples[0].get('context') is not None:
                prompt = sample.get('context') + ' ' + prompt

            prompts.append(prompt)

        self.tokenizer.padding_side = 'left'
        input_text = self.tokenizer(
            prompts,
            padding='longest',
            truncation=True,
            return_tensors='pt',
            max_length=2000,
        ).to(device)
        return input_text

    def forward(self, batch: dict) -> Union[DataSample, List[DataSample]]:

        if self.mode == 'generation':
            return self.generate(batch)
        else:
            raise RuntimeError(f'Unsupported mode: {self.mode}')

    def generate(self, batch: dict) -> Union[DataSample, List[DataSample]]:
        batch = self.data_preprocessor(batch, False)
        images = batch['images']
        data_samples = batch['data_samples']
        return self.predict(images, data_samples)
