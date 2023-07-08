from typing import List

import torch
import torch.nn as nn
from mmengine.device import get_device
from open_flamingo import create_model_and_transforms

from opencompass.registry import MM_MODELS


@MM_MODELS.register_module('openflamingov2-omnimmbench')
class OpenFlamingoV2OmniMMBench(nn.Module):
    def __init__(
        self,
        ckpt_path,
        clip_vision_encoder_path,
        clip_vision_encoder_pretrained,
        lang_encoder_path,
        tokenizer_path,
        cross_attn_every_n_layers,
    ) -> None:
        super().__init__()

        model, _, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=clip_vision_encoder_path,
            clip_vision_encoder_pretrained=clip_vision_encoder_pretrained,
            lang_encoder_path=lang_encoder_path,
            tokenizer_path=tokenizer_path,
            cross_attn_every_n_layers=cross_attn_every_n_layers)

        msg = model.load_state_dict(torch.load(ckpt_path), strict=False)

        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'

    def _prepare_images(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        """Preprocess images and stack them.

        Args:
            batch: A list of lists of images. (frames default to 1)

        Returns:
            A Tensor of shape
            (batch_size, images_per_example, frames, channels, height, width).
        """
        images_per_example = max(len(x) for x in batch)
        batch_images = None
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                # preprocessed = self.image_processor(image)
                preprocessed = image
                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) +
                        preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        return batch_images

    def process_input(self, batch):
        # parse input
        # batch_size x num_media x num_frames x channels x height x width
        # [B, 1, 1, C, H, W]
        image = batch.pop('inputs')
        image = [[im] for im in image]
        image = self._prepare_images(image)

        data_samples = batch.pop('data_samples')

        # generate text prompt
        prompt = [
            '<image>{} {}Answer:'.format(ds.question, ds.options)
            for ds in data_samples
        ]

        return image, prompt, data_samples

    def generate(self, batch):
        image, prompt, data_sample = self.process_input(batch)

        text_encoded = self.tokenizer(prompt,
                                      padding='longest',
                                      max_length=2000,
                                      return_tensors='pt')
        input_ids = text_encoded['input_ids']
        attention_mask = text_encoded['attention_mask']

        with torch.inference_mode():
            output = self.model.generate(
                vision_x=image.to(get_device()),
                lang_x=input_ids.to(get_device()),
                attention_mask=attention_mask.to(get_device()),
                max_new_tokens=50,
                num_beams=5,
            )

        answer = output[:, len(input_ids[0]):]
        result = self.tokenizer.batch_decode(answer, skip_special_tokens=True)
        result = [str.strip(r) for r in result]

        for i, sample in enumerate(data_sample):
            data_sample[i].pred_answer = result[i]

        return data_sample
