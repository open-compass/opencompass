# Copyright (c) OpenMMLab. All rights reserved.
"""Taken from https://github.com/lucidrains/flamingo-pytorch."""

from typing import Optional

import torch
from einops import rearrange, repeat
from torch import einsum, nn


def FeedForward(dim, mult: int = 4):
    """Feedforward layers.

    Args:
        mult (int): Layer expansion muliplier. Defaults to 4.
    """
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    """Perceiver attetion layers.

    Args:
        dim (int): Input dimensions.
        dim_head (int): Number of dimension heads. Defaults to 64.
        heads (int): Number of heads. Defaults to 8.
    """
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor):
        """Forward function.

        Args:
            x (torch.Tensor): image features of shape (b, T, n1, D).
            latent (torch.Tensor): latent features of shape (b, T, n2, D).
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q = rearrange(q, 'b t n (h d) -> b h t n d', h=h)
        k = rearrange(k, 'b t n (h d) -> b h t n d', h=h)
        v = rearrange(v, 'b t n (h d) -> b h t n d', h=h)
        q = q * self.scale

        # attention
        sim = einsum('... i d, ... j d  -> ... i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """Perceiver resampler layers.

    Args:
        dim (int): Input dimensions.
        depth (int): Depth of resampler. Defaults to 6.
        dim_head (int): Number of dimension heads. Defaults to 64.
        heads (int): Number of heads. Defaults to 8.
        num_latents (int): Number of latents. Defaults to 64.
        max_num_media (int, optional): Max number of media.
            Defaults to None.
        max_num_frames (int, optional): Max number of frames.
            Defaults to None.
        ff_mult (int): Feed forward multiplier. Defaults to 4.
    """
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 6,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        max_num_media: Optional[int] = None,
        max_num_frames: Optional[int] = None,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = (nn.Parameter(torch.randn(max_num_frames, dim))
                           if max_num_frames is not None else None)
        self.media_time_embs = (nn.Parameter(torch.randn(
            max_num_media, 1, dim)) if max_num_media is not None else None)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim=dim, dim_head=dim_head,
                                       heads=heads),
                    FeedForward(dim=dim, mult=ff_mult),
                ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        """Forward function for perceiver sampler.

        Args:
            x (torch.Tensor): image features of shape (b, T, F, v, D)

        Returns:
            torch.Tensor: shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if self.frame_embs is not None:
            frame_embs = repeat(self.frame_embs[:F],
                                'F d -> b T F v d',
                                b=b,
                                T=T,
                                v=v)
            x = x + frame_embs
        x = rearrange(x, 'b T F v d -> b T (F v) d'
                      )  # flatten the frame and spatial dimensions
        if self.media_time_embs is not None:
            x = x + self.media_time_embs[:T]

        # blocks
        latents = repeat(self.latents, 'n d -> b T n d', b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


class MaskedCrossAttention(nn.Module):
    """Masked cross attention layers.

    Args:
        dim (int): Input text feature dimensions.
        dim_visual (int): Input visual feature dimensions.
        dim_head (int): Number of dimension heads. Defaults to 64.
        heads (int): Number of heads. Defaults to 8.
        only_attend_immediate_media (bool): Whether attend immediate media.
            Defaults to True.
    """
    def __init__(
        self,
        *,
        dim: int,
        dim_visual: int,
        dim_head: int = 64,
        heads: int = 8,
        only_attend_immediate_media: bool = True,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image
        # or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self,
                x: torch.Tensor,
                media: torch.Tensor,
                media_locations: Optional[torch.Tensor] = None,
                attend_previous: bool = True):
        """Forward function for perceiver sampler.

        Args:
            x (torch.Tensor): text features of shape (B, T_txt, D_txt).
            media (torch.Tensor): image features of shape
                (B, T_img, n, D_img) where n is the dim of the latents.
            media_locations (torch.Tensor, optional): boolean mask identifying
                the media tokens in x of shape (B, T_txt). Defaults to None.
            attend_previous (bool): If false, ignores immediately preceding
                image and starts attending when following image.
                Defaults to True.
        """
        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, 'b t n d -> b (t n) d')

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)

        if media_locations is not None:
            # at each boolean of True, increment the time counter
            # (relative to media time)
            text_time = media_locations.cumsum(dim=-1)
            media_time = torch.arange(T_img, device=x.device) + 1

            if not attend_previous:
                text_time[~media_locations] += 1
                # make sure max is still the number of images in the sequence
                text_time[text_time > repeat(
                    torch.count_nonzero(media_locations, dim=1),
                    'b -> b i',
                    i=text_time.shape[1],
                )] = 0

            # text time must equal media time if only attending to most
            # immediate image otherwise, as long as text time is greater than
            # media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge  # noqa

            text_to_media_mask = mask_op(
                rearrange(text_time, 'b i -> b 1 i 1'),
                repeat(media_time, 'j -> 1 1 1 (j n)', n=n),
            )
            sim = sim.masked_fill(~text_to_media_mask,
                                  -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if media_locations is not None and self.only_attend_immediate_media:
            # any text without a preceding media needs to have
            # attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask,
                                                'b i -> b 1 i 1')
            attn = attn.masked_fill(text_without_media_mask, 0.0)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    """Gated cross attention layers.

    Args:
        dim (int): Input text feature dimensions.
        dim_visual (int): Input visual feature dimensions.
        dim_head (int): Number of dimension heads. Defaults to 64.
        heads (int): Number of heads. Defaults to 8.
        ff_mult (int): Feed forward multiplier. Defaults to 4.
        only_attend_immediate_media (bool): Whether attend immediate media.
            Defaults to True.
    """
    def __init__(
        self,
        *,
        dim: int,
        dim_visual: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        only_attend_immediate_media: bool = True,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self,
                x: torch.Tensor,
                media: torch.Tensor,
                media_locations: Optional[torch.Tensor] = None,
                attend_previous: bool = True):
        """Forward function for perceiver sampler.

        Args:
            x (torch.Tensor): text features of shape (B, T_txt, D_txt).
            media (torch.Tensor): image features of shape
                (B, T_img, n, D_img) where n is the dim of the latents.
            media_locations (torch.Tensor, optional): boolean mask identifying
                the media tokens in x of shape (B, T_txt). Defaults to None.
            attend_previous (bool): If false, ignores immediately preceding
                image and starts attending when following image.
                Defaults to True.
        """
        x = (self.attn(
            x,
            media,
            media_locations=media_locations,
            attend_previous=attend_previous,
        ) * self.attn_gate.tanh() + x)
        x = self.ff(x) * self.ff_gate.tanh() + x

        return x


class FlamingoLayer(nn.Module):
    """Faminogo layers.

    Args:
        gated_cross_attn_layer (nn.Module): Gated cross attention layer.
        decoder_layer (nn.Module): Decoder layer.
    """
    def __init__(self, gated_cross_attn_layer: nn.Module,
                 decoder_layer: nn.Module):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None

    def condition_vis_x(self, vis_x):
        """Set condition vision features."""
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations):
        """Set condition media locations."""
        self.media_locations = media_locations

    def condition_attend_previous(self, attend_previous):
        """Set attend previous."""
        self.attend_previous = attend_previous

    def forward(
        self,
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **decoder_layer_kwargs,
    ):
        """Forward function.

        Args:
            lang_x (torch.Tensor): language inputs.
            attention_mask (torch.Tensor, optional): text attention mask.
                Defaults to None.
            **decoder_layer_kwargs: Other decoder layer keyword arguments.
        """
        if self.gated_cross_attn_layer is None:
            return self.decoder_layer(lang_x,
                                      attention_mask=attention_mask,
                                      **decoder_layer_kwargs)

        if self.vis_x is None:
            raise ValueError('vis_x must be conditioned before forward pass')

        if self.media_locations is None:
            raise ValueError(
                'media_locations must be conditioned before forward pass')

        lang_x = self.gated_cross_attn_layer(
            lang_x,
            self.vis_x,
            media_locations=self.media_locations,
            attend_previous=self.attend_previous,
        )
        lang_x = self.decoder_layer(lang_x,
                                    attention_mask=attention_mask,
                                    **decoder_layer_kwargs)
        return lang_x
