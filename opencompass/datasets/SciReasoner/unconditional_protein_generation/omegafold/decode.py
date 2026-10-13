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
For generating the final coordinates of the amino acids of the predicted
"""
# =============================================================================
# Imports
# =============================================================================
import argparse
import math
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


class InvariantPointAttention(modules.OFModule):
    """
    This is the Invariant Point Attention from Jumper et al. (2021) that
    performs transformer-like operation on frames

    """

    def __init__(self, cfg: argparse.Namespace) -> None:
        super(InvariantPointAttention, self).__init__(cfg)
        node_dim = cfg.node_dim
        edge_dim = cfg.edge_dim
        num_head = cfg.num_head
        num_scalar_qk = cfg.num_scalar_qk
        num_point_qk = cfg.num_point_qk
        num_scalar_v = cfg.num_scalar_v
        num_point_v = cfg.num_point_v

        # For scalar parts
        self.q_scalar = nn.Linear(node_dim, num_head * num_scalar_qk)
        self.k_scalar = nn.Linear(node_dim, num_head * num_scalar_qk)
        self.v_scalar = nn.Linear(node_dim, num_head * num_scalar_v)

        # to reason about the spatial relationships
        self.q_point = nn.Linear(node_dim, num_head * 3 * num_point_qk)
        self.k_point = nn.Linear(node_dim, num_head * 3 * num_point_qk)
        self.v_point = nn.Linear(node_dim, num_head * 3 * num_point_v)

        # trainable weights for edge bias
        self.trainable_point_weights = nn.Parameter(
            torch.full([cfg.num_head],
                       fill_value=math.log(math.exp(1.) - 1)), )
        self.bias_2d = nn.Linear(edge_dim, num_head)

        final_input_dim = edge_dim + num_scalar_v + num_point_v * 4
        final_input_dim *= num_head
        # output projection
        self.output_projection = nn.Linear(final_input_dim, node_dim)
        self.softplus = torch.nn.Softplus()

        # weighting of each component
        num_logit_terms = 3
        scalar_variance = max(num_scalar_qk, 1) * 1.
        point_variance = max(num_point_qk, 1) * 9. / 2
        self.scalar_weight = math.sqrt(1 / (num_logit_terms * scalar_variance))
        self.point_weight = math.sqrt(1 / (num_logit_terms * point_variance))
        self.edge_logits_weight = math.sqrt(1 / num_logit_terms)

    def forward(self, node_repr: torch.Tensor, edge_repr: torch.Tensor,
                frames: utils.AAFrame) -> torch.Tensor:
        """
        From Jumper et al. (2021), Invariant Point Attention

        Args:
            node_repr: the node representation,
                of shape [num_res, dim_node]
            edge_repr: the edge representation,
                of shape [num_res, num_res, dim_edge]
            frames: the backbone frames of the amino acids,
                of shape [num_res]

        Returns:
            the node representation update of shape [num_res, dim_node]

        """
        n_head = self.cfg.num_head

        # acquire the scalar part of the attention logits
        _q_scalar = self._get_scalar(self.q_scalar, node_repr, n_head)
        _k_scalar = self._get_scalar(self.k_scalar, node_repr, n_head)
        _v_scalar = self._get_scalar(self.v_scalar, node_repr, n_head)
        scalar_logits = torch.einsum('qhc,khc->qkh', _q_scalar, _k_scalar)
        scalar_logits *= self.scalar_weight

        # acquire the 2-dimensional bias from the edge representation
        edge_logits = self.bias_2d(edge_repr) * self.edge_logits_weight

        # acquire the spatial part of the logits from the frames
        _q_point = self._get_point(self.q_point, node_repr, n_head, frames)
        _k_point = self._get_point(self.k_point, node_repr, n_head, frames)
        _v_point = self._get_point(self.v_point, node_repr, n_head, frames)
        dist = ((_q_point[:, None, ...] - _k_point[None, ...])**2)
        point_logits = dist.sum([-1, -2]) * self.point_weight
        point_logits *= self.softplus(self.trainable_point_weights) / 2

        # Combine them and take the softmax
        logits = scalar_logits + edge_logits - point_logits
        logits += utils.mask2bias(frames.mask[None, ..., None])
        attn_w = modules.softmax(logits, dim=-2, in_place=True)

        # get the output
        ret_edge = torch.einsum('...qkh,...qkc->...qhc', attn_w, edge_repr)
        ret_scalar = torch.einsum('...qkh,...khc->...qhc', attn_w, _v_scalar)
        ret_point = torch.einsum('...qkh,...khpc->...qhpc', attn_w, _v_point)
        ret_point = frames.position_in_frame(ret_point)

        feat = torch.cat([
            ret_scalar.flatten(start_dim=-2),
            ret_point.flatten(start_dim=-3),
            utils.get_norm(ret_point).flatten(start_dim=-2),
            ret_edge.flatten(start_dim=-2),
        ],
                         dim=-1)

        return self.output_projection(feat)

    @staticmethod
    def _get_scalar(linear: nn.Linear, inputs: torch.Tensor,
                    num_head: int) -> torch.Tensor:
        """
        Pass the input through linear and then perform reshaping for the
        multi-headed attention

        Args:
            linear: the linear module to pass the input into
            inputs: the input tensor to the linear module
            num_head: the number of heads

        Returns:
            key, query, or value for the multi-headed attention,
                [num_res, num_head, dim]

        """
        return linear(inputs).unflatten(dim=-1, sizes=[num_head, -1])

    @staticmethod
    def _get_point(linear: nn.Linear, inputs: torch.Tensor, n_head: int,
                   transformation: utils.AAFrame) -> torch.Tensor:
        """
        Pass the input through the linear and perform reshaping for the
        multi-headed attention, then transform the points by the transformation

        Args:
            linear: the linear module to compute the local points
            inputs: the inputs into the linear module, of shape
            n_head: the number of head
            transformation: the transformation to make local global

        Returns:
            points in global frame, [num_res, n_head, -1, 3]

        """
        local_points = linear(inputs).unflatten(dim=-1, sizes=[n_head, -1, 3])
        global_points = transformation.transform(local_points)
        return global_points


class TorsionAngleHead(modules.OFModule):
    """
    Predict the torsion angles of each of the amino acids from
    node representation following Jumper et al. (2021)
    """

    def __init__(self, cfg: argparse.Namespace):
        super(TorsionAngleHead, self).__init__(cfg)

        self.input_projection = nn.ModuleList(
            [nn.Linear(cfg.node_dim, cfg.num_channel) for _ in range(2)])

        self.resblock1 = nn.ModuleList([
            nn.Linear(cfg.num_channel, cfg.num_channel)
            for _ in range(cfg.num_residual_block)
        ])
        self.resblock2 = nn.ModuleList([
            nn.Linear(cfg.num_channel, cfg.num_channel)
            for _ in range(cfg.num_residual_block)
        ])

        self.unnormalized_angles = nn.Linear(cfg.num_channel, 14)

    def forward(
            self, representations_list: typing.Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """
        Predict side chains using multi-rigid representations.

        Args:
            representations_list: A list of activations to
            predict side chains from.
        Returns:
            The normalized sin-cos representation of the torsion angles
        """
        act = 0.
        for (x, layer) in zip(representations_list, self.input_projection):
            act = layer(torch.relu(x)) + act

        for layer1, layer2 in zip(self.resblock1, self.resblock2):
            old_act = act
            act = layer1(torch.relu(act))
            act = layer2(torch.relu(act))
            act = old_act + act

        sin_cos_raw = self.unnormalized_angles(torch.relu(act))

        sin_cos_raw = sin_cos_raw.unflatten(dim=-1, sizes=[7, 2])
        sin_cos_normalized = utils.robust_normalize(sin_cos_raw)

        return sin_cos_normalized


class StructureCycle(modules.OFModule):
    """
    Each of the cycles from
        Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"

    """

    def __init__(self, cfg: argparse.Namespace) -> None:
        super(StructureCycle, self).__init__(cfg)
        self.ipa = InvariantPointAttention(cfg)
        self.input_norm = nn.LayerNorm(cfg.node_dim)
        self.transition = nn.ModuleList([
            nn.Linear(cfg.node_dim, cfg.node_dim)
            for _ in range(cfg.num_transition)
        ])
        self.update_norm = nn.LayerNorm(cfg.node_dim)

        self.affine_update = nn.Linear(cfg.node_dim, 6)

    def forward(
        self, node_repr: torch.Tensor, edge_repr: torch.Tensor,
        backbone_frames: utils.AAFrame
    ) -> typing.Tuple[torch.Tensor, utils.AAFrame]:
        """
        Perform one backbone update and node representation update

        Args:
            node_repr: the node representation,
                of shape [num_res, dim_node]
            edge_repr: the edge representation,
                of shape [num_res, dim_edge]
            backbone_frames: the backbone frames of the amino acids,
                of shape [num_res]

        Returns:

        """
        node_repr += self.ipa(node_repr, edge_repr, backbone_frames)
        node_repr = self.input_norm(node_repr)
        # Transition
        input_repr = node_repr
        for layer in self.transition:
            node_repr = layer(node_repr)
            if layer is not self.transition[-1]:
                node_repr = torch.relu(node_repr)

        node_repr += input_repr  # Shortcut residual connection
        node_repr = self.update_norm(node_repr)
        backbone_update = self.affine_update(node_repr)
        frame_update = utils.AAFrame.from_tensor(backbone_update, unit='nano')
        backbone_frames = backbone_frames * frame_update

        return node_repr, backbone_frames


class StructureModule(modules.OFModule):
    """Jumper et al. (2021) Suppl. Alg. 20 'StructureModule'"""

    def __init__(self, cfg: argparse.Namespace):
        super(StructureModule, self).__init__(cfg)
        self.node_norm = nn.LayerNorm(cfg.node_dim)
        self.edge_norm = nn.LayerNorm(cfg.edge_dim)
        self.init_proj = nn.Linear(cfg.node_dim, cfg.node_dim)

        self.cycles = nn.ModuleList(
            [StructureCycle(cfg) for _ in range(cfg.num_cycle)])

        self.torsion_angle_pred = TorsionAngleHead(cfg)

    def forward(
        self, node_repr: torch.Tensor, edge_repr: torch.Tensor,
        fasta: torch.Tensor, mask: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, typing.Dict[str, typing.Union[
            utils.AAFrame, torch.Tensor]]]:
        """
        Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"

        Args:
            node_repr: node representation tensor of shape [num_res, dim_node]
            edge_repr: edge representation tensor of shape [num_res, dim_edge]
            fasta: the tokenized sequence of the input protein sequence
            mask

        Returns:
            node_repr: The current node representation tensor for confidence
                of shape [num_res, dim_node]
            dictionary containing:
                final_atom_positions: the final atom14 positions,
                    of shape [num_res, 14, 3]
                final_atom_mask: the final atom14 mask,
                    of shape [num_res, 14]

        """
        node_repr = self.node_norm(node_repr)
        edge_repr = self.edge_norm(edge_repr)

        init_node_repr = node_repr
        node_repr = self.init_proj(node_repr)
        # Initialize the initial frames with Black-hole Jumper et al. (2021)
        backbone_frames = utils.AAFrame.default_init(*node_repr.shape[0:1],
                                                     unit='nano',
                                                     device=self.device,
                                                     mask=mask.bool())

        for layer in self.cycles:
            node_repr, backbone_frames = layer(node_repr, edge_repr,
                                               backbone_frames)

        torsion_angles_sin_cos = self.torsion_angle_pred(
            representations_list=[node_repr, init_node_repr], )

        torsion_angles_mask = torch.ones_like(torsion_angles_sin_cos[..., 0],
                                              dtype=torch.bool)
        backbone_frames = backbone_frames.to_angstrom(in_place=False)
        frames8 = backbone_frames.expand_w_torsion(
            torsion_angles=torsion_angles_sin_cos,
            torsion_angles_mask=torsion_angles_mask,
            fasta=fasta)
        pos14, mask14 = frames8.expanded_to_pos(fasta)
        return node_repr, {
            'final_frames': frames8,
            'final_atom_positions': pos14,
            'final_atom_mask': mask14
        }


# =============================================================================
# Tests
# =============================================================================
if __name__ == '__main__':
    pass
