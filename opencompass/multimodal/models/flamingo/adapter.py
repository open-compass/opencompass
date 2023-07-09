# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch.nn as nn

from opencompass.registry import MM_MODELS

from .modules import FlamingoLayer, GatedCrossAttentionBlock
from .utils import getattr_recursive, setattr_recursive


@MM_MODELS.register_module()
class FlamingoLMAdapter:
    """Mixin to add cross-attention layers to a language model."""
    @classmethod
    def extend_init(
        cls,
        base: object,
        vis_hidden_size: int,
        cross_attn_every_n_layers: int,
        use_media_placement_augmentation: bool,
    ):
        """Initialize Flamingo by adding a new gated cross attn to the decoder.

        Store the media token id for computing the media locations.

        Args:
            base (object): Base module could be any object that represent
                a instance of language model.
            vis_hidden_size: (int): Hidden size of vision embeddings.
            cross_attn_every_n_layers: (int): Additional cross attn for
                every n layers.
            use_media_placement_augmentation: (bool): Whether to use media
                placement augmentation.
        """
        base.set_decoder_layers_attr_name('model.layers')
        gated_cross_attn_layers = nn.ModuleList([
            GatedCrossAttentionBlock(dim=base.config.hidden_size,
                                     dim_visual=vis_hidden_size) if
            (layer_idx + 1) % cross_attn_every_n_layers == 0 else None
            for layer_idx, _ in enumerate(base._get_decoder_layers())
        ])
        base._set_decoder_layers(
            nn.ModuleList([
                FlamingoLayer(gated_cross_attn_layer, decoder_layer)
                for gated_cross_attn_layer, decoder_layer in zip(
                    gated_cross_attn_layers, base._get_decoder_layers())
            ]))
        base.use_media_placement_augmentation = use_media_placement_augmentation  # noqa
        base.initialized_flamingo = True
        return base

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        """Set decoder layers attribute name."""
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        """Get decoder layers according to attribute name."""
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        """Set decoder layers according to attribute name."""
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def forward(self, *input, **kwargs):
        """Condition the Flamingo layers on the media locations before forward
        function."""
        input_ids = kwargs['input_ids'] if 'input_ids' in kwargs else input[0]
        media_locations = input_ids == self.media_token_id
        attend_previous = ((random.random() < 0.5)
                           if self.use_media_placement_augmentation else False)

        for layer in self.get_decoder().layers:
            layer.condition_media_locations(media_locations)
            layer.condition_attend_previous(attend_previous)

        return super().forward(
            *input, **kwargs)  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(layer.is_conditioned()
                   for layer in self._get_decoder_layers())

    def clear_conditioned_layers(self):
        """Clear all conditional layers."""
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_attend_previous(None)
