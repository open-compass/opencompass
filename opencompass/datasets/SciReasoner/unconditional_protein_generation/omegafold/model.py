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

"""
# =============================================================================
# Imports
# =============================================================================
import argparse
import typing

import torch
from torch import nn

from . import (confidence, decode, embedders, geoformer, modules, omegaplm,
               utils)
from .utils import residue_constants as rc

# =============================================================================
# Constants
# =============================================================================
# =============================================================================
# Functions
# =============================================================================
# =============================================================================
# Classes
# =============================================================================


class OmegaFoldCycle(modules.OFModule):

    def __init__(self, cfg: argparse.Namespace) -> None:
        super(OmegaFoldCycle, self).__init__(cfg)

        self.geoformer = geoformer.GeoFormer(cfg)
        self.structure_module = decode.StructureModule(cfg.struct)
        self.confidence_head = confidence.ConfidenceHead(cfg.struct)

    def forward(
        self,
        fasta: torch.Tensor,
        mask: torch.Tensor,
        node_repr: torch.Tensor,
        edge_repr: torch.Tensor,
        fwd_cfg: typing.Optional[argparse.Namespace],
    ) -> typing.Tuple[typing.Dict[str, torch.Tensor], typing.Dict[
            str, typing.Union[torch.Tensor, utils.AAFrame]]]:
        """
        The forward method for one iteration of OmegaFold

        Args:
            fasta: the tokenized sequence of the protein, of shape,
                of shape [num_res]
            mask: If to ignore, of shape,
                of shape [num_res]
            node_repr:
                of shape [num_res, node_repr_dim]
            edge_repr:
                of shape [num_res, node_repr, edge_repr_dim]
            fwd_cfg:

        Returns:
            ret: A dictionary containing:
            confidence: the confidence score of the output protein structure

        """

        prev_node, edge_repr, node_repr = self.geoformer(node_repr=node_repr,
                                                         edge_repr=edge_repr,
                                                         mask=mask,
                                                         fwd_cfg=fwd_cfg)

        node_repr, ret = self.structure_module(
            node_repr=node_repr[..., 0, :, :],
            edge_repr=edge_repr,
            fasta=fasta,
            mask=mask[..., 0, :],
        )

        ret['confidence'] = self.confidence_head(node_repr)

        prev_dict = {
            'prev_node': prev_node[..., 0, :, :],
            'prev_edge': edge_repr,
            'prev_x': ret['final_atom_positions'],
            'prev_frames': ret['final_frames'],
        }
        return ret, prev_dict


_INPUTS = typing.List[typing.Dict[typing.Union[str, int], typing.Any]]


class OmegaFold(modules.OFModule):
    """
    The Entire OmegaFold model that comprises a pretrained Protein Language
    Model, an encoder of the primary sequence, as well as a structure module
    for decoding

    """

    def __init__(self, cfg: argparse.Namespace) -> None:
        super(OmegaFold, self).__init__(cfg)
        self.omega_plm = omegaplm.OmegaPLM(cfg.plm)
        self.plm_node_embedder = nn.Linear(cfg.plm.node, cfg.node_dim)
        self.plm_edge_embedder = nn.Linear(cfg.plm.edge, cfg.edge_dim)
        self.input_embedder = embedders.EdgeEmbedder(cfg)
        self.recycle_embedder = embedders.RecycleEmbedder(cfg)
        self.omega_fold_cycle = OmegaFoldCycle(cfg)

    def forward(
        self,
        inputs: _INPUTS,
        predict_with_confidence: typing.Optional[bool] = True,
        *,
        fwd_cfg: typing.Optional[argparse.Namespace] = None
    ) -> typing.Dict[str, typing.Union[torch.Tensor, float]]:
        """
        The forward implementation of OmegaFold

        Args:
            inputs:
            predict_with_confidence: if to choose with confidence
            fwd_cfg: forward configuration

        Returns:

        """
        # Preparation before entering the cycles
        primary_sequence = inputs[0]['p_msa'][..., 0, :]
        max_confidence = 0
        prev_dict = self.create_initial_prev_dict(len(primary_sequence))
        final_result = None

        # Start cycling
        residx_atom14_mask = rc.restype_atom14_mask.to(
            device=primary_sequence.device)[primary_sequence]
        for cycle_data in inputs:
            p_msa, p_msa_mask = cycle_data['p_msa'], cycle_data['p_msa_mask']
            fasta, mask = p_msa[..., 0, :], p_msa_mask[..., 0, :]
            node_repr, edge_repr = self.deep_sequence_embed(
                p_msa, p_msa_mask, fwd_cfg)
            node_recycle, edge_repr = self.recycle_embedder(
                fasta=fasta,
                prev_node=prev_dict.pop('prev_node'),
                prev_edge=prev_dict.pop('prev_edge'),
                prev_x=prev_dict.pop('prev_x'),
                node_repr=node_repr,
                edge_repr=edge_repr,
                atom14_mask=residx_atom14_mask,
                prev_frames=prev_dict.pop('prev_frames'))

            result, prev_dict = self.omega_fold_cycle(fasta=fasta,
                                                      mask=p_msa_mask,
                                                      node_repr=node_repr,
                                                      edge_repr=edge_repr,
                                                      fwd_cfg=fwd_cfg)

            confidence_overall = confidence.get_all_confidence(
                result['confidence'],
                result['final_atom_positions'][..., 1, :], mask)
            result['confidence_overall'] = confidence_overall
            if predict_with_confidence:
                if confidence_overall > max_confidence:
                    max_confidence = confidence_overall
                    final_result = result
            else:
                final_result = result

        return final_result

    def deep_sequence_embed(
        self,
        fasta: torch.Tensor,
        mask: torch.Tensor,
        fwd_cfg: typing.Optional[argparse.Namespace],
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the forward method of the pretrained-language model

        Args:
            fasta: the fasta sequence
            mask: the mask indicating the validity of the token

        Returns:

        """
        node_repr, edge_repr = self.omega_plm(fasta, mask, fwd_cfg=fwd_cfg)
        # return node_plm, edge_plm
        node_repr = self.plm_node_embedder(
            utils.normalize(node_repr, in_place=True))
        edge_repr = edge_repr.permute(1, 2, 0)
        edge_repr = self.plm_edge_embedder(
            utils.normalize(edge_repr, in_place=True))
        edge_repr = self.input_embedder(fasta[..., 0, :], out=edge_repr)

        return node_repr, edge_repr

    def create_initial_prev_dict(
            self, num_res: int) -> typing.Dict[str, torch.Tensor]:
        """
        Generate 'previous' (filling with 0's) features for the model

        Args:
            num_res: the number of residues

        Returns:

        """
        return {
            'prev_node':
            torch.zeros([num_res, self.cfg.node_dim],
                        device=self.device,
                        dtype=torch.float),
            'prev_edge':
            torch.zeros([num_res, num_res, self.cfg.edge_dim],
                        device=self.device,
                        dtype=torch.float),
            'prev_x':
            torch.zeros([num_res, 14, 3],
                        device=self.device,
                        dtype=torch.float),
            'prev_frames':
            utils.AAFrame.default_init(num_res,
                                       8,
                                       unit='Angstrom',
                                       device=self.device)
        }


# =============================================================================
# Tests
# =============================================================================
if __name__ == '__main__':
    pass
