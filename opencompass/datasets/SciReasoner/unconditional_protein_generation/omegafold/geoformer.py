# =============================================================================
# Copyright 2022 HeliXon Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
The code for GeoFormer, the main trunk
"""
# =============================================================================
# Imports
# =============================================================================
import argparse
import typing

import torch
from torch import nn

from . import modules, utils

# =============================================================================
# Constants
# =============================================================================
# =============================================================================
# Functions
# =============================================================================
# =============================================================================
# Classes
# =============================================================================


class GeoFormerBlock(modules.OFModule):
    """
    One iteration of GeoFormer

    """

    def __init__(self, cfg: argparse.Namespace) -> None:
        super(GeoFormerBlock, self).__init__(cfg)
        self.attention_w_edge_bias = modules.AttentionWEdgeBias(
            d_node=cfg.node_dim,
            d_edge=cfg.edge_dim,
            n_head=cfg.attn_n_head,
            attn_gating=cfg.gating,
            attn_c=cfg.attn_c)
        self.column_attention = modules.Attention(q_dim=cfg.node_dim,
                                                  kv_dim=cfg.node_dim,
                                                  gating=cfg.gating,
                                                  n_head=cfg.attn_n_head,
                                                  c=cfg.attn_c,
                                                  out_dim=cfg.node_dim,
                                                  n_axis=1)
        self.node_transition = modules.Transition(d=cfg.node_dim,
                                                  n=cfg.transition_multiplier,
                                                  activation=cfg.activation)
        self.out_product = modules.Node2Edge(in_dim=cfg.node_dim,
                                             out_dim=cfg.edge_dim,
                                             proj_dim=cfg.opm_dim)
        self.geometric_attention = nn.ModuleList([
            modules.GeometricAttention(d_edge=cfg.edge_dim,
                                       n_axis=2,
                                       c=cfg.geom_c,
                                       n_head=cfg.geom_head)
            for _ in range(cfg.geom_count)
        ])
        self.edge_transition = modules.Transition(d=cfg.edge_dim,
                                                  n=cfg.transition_multiplier,
                                                  activation=cfg.activation)

    def forward(
        self,
        node_repr: torch.Tensor,
        edge_repr: torch.Tensor,
        mask: torch.Tensor,
        *,
        fwd_cfg: typing.Optional[argparse.Namespace] = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            node_repr:
            edge_repr:
            mask
            fwd_cfg:

        Returns:

        """
        node_repr += self.attention_w_edge_bias(node_repr,
                                                edge_repr,
                                                mask,
                                                fwd_cfg=fwd_cfg)
        node_repr = self._column_attention(node_repr, mask, fwd_cfg=fwd_cfg)
        node_repr += self.node_transition(node_repr,
                                          subbatch_size=fwd_cfg.subbatch_size)

        edge_repr += self.out_product(node_repr, mask)
        for layer in self.geometric_attention:
            edge_repr += layer(edge_repr, mask[..., 0, :], fwd_cfg=fwd_cfg)

        edge_repr += self.edge_transition(edge_repr, fwd_cfg.subbatch_size)

        return node_repr, edge_repr

    def _column_attention(self, node_repr, mask, fwd_cfg):
        node_repr_col = utils.normalize(
            node_repr.transpose(-2, -3).contiguous())
        node_repr_col = self.column_attention(node_repr_col,
                                              node_repr_col,
                                              bias=utils.mask2bias(
                                                  mask.T[..., None, None, :]),
                                              fwd_cfg=fwd_cfg)
        node_repr += node_repr_col.transpose(-2, -3)
        return node_repr


class GeoFormer(modules.OFModule):

    def __init__(self, cfg: argparse.Namespace):
        super(GeoFormer, self).__init__(cfg)
        self.blocks = nn.ModuleList(
            [GeoFormerBlock(cfg) for _ in range(cfg.geo_num_blocks)])
        self.node_final_proj = nn.Linear(cfg.node_dim, cfg.struct.node_dim)

    def forward(
        self,
        node_repr: torch.Tensor,
        edge_repr: torch.Tensor,
        mask: torch.Tensor,
        *,
        fwd_cfg: typing.Optional[argparse.Namespace] = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            node_repr: the node representation from the
                pretrained language model, of shape[num_res, dim]
            edge_repr: the edge representation from the
                pretrained language model, of shape[num_res, num_res, dim]
            mask: the mask indicating the validity of the amino acid,
                of [num_res].
            fwd_cfg

        Returns:
            edge_repr: the edge representation used for recycling
            node_repr: the node representation used for recycling
            final_node: the node representation used for structure generation

        """

        for block in self.blocks:
            node_repr, edge_repr = block(node_repr,
                                         edge_repr,
                                         mask,
                                         fwd_cfg=fwd_cfg)

        final_node = self.node_final_proj(node_repr)
        return node_repr, edge_repr, final_node


# =============================================================================
# Tests
# =============================================================================
if __name__ == '__main__':
    pass
