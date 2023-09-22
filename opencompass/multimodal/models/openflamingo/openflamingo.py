import re
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
                 prompt_constructor: dict,
                 post_processor: Optional[dict] = None,
                 mode: str = 'generation',
                 **kwargs):
        super().__init__(**kwargs)
        self.prompt_constructor = mmengine.registry.build_from_cfg(
            prompt_constructor, MM_MODELS)
        if post_processor is not None:
            self.post_processor = mmengine.registry.build_from_cfg(
                post_processor, MM_MODELS)
        else:
            self.post_processor = None
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
        prompts = self.prompt_constructor(data_samples)

        self.tokenizer.padding_side = 'left'
        input_text = self.tokenizer(
            prompts,
            padding='longest',
            truncation=True,
            return_tensors='pt',
            max_length=2000,
        ).to(device)
        return input_text

    def post_process(
            self, outputs: torch.Tensor,
            data_samples: Optional[List[DataSample]]) -> List[DataSample]:
        """Perform post process for outputs for different task.

        Args:
            outputs (torch.Tensor): The generated outputs.
            data_samples (List[DataSample], optional): The annotation
                data of every samples.

        Returns:
            List[DataSample]: Return list of data samples.
        """
        outputs = self.tokenizer.batch_decode(outputs,
                                              skip_special_tokens=True)

        if data_samples is None:
            data_samples = [DataSample() for _ in range(len(outputs))]

        for output, data_sample in zip(outputs, data_samples):
            # remove text pattern
            if self.task == 'caption':
                data_sample.pred_caption = re.split('Output', output,
                                                    1)[0].replace('"', '')
                if self.post_processor:
                    data_sample.pred_caption = self.post_processor(
                        data_sample.pred_caption)
            elif self.task == 'vqa':
                data_sample.pred_answer = re.split('Question|Answer', output,
                                                   1)[0]
                if self.post_processor:
                    data_sample.pred_answer = self.post_processor(
                        data_sample.pred_answer)

        return data_samples

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
