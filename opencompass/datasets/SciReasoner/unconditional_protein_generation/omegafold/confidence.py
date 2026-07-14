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
Code for confidence-relevant things
"""

# =============================================================================
# Imports
# =============================================================================
import argparse

import torch
from torch import nn

from . import modules, utils

# =============================================================================
# Constants
# =============================================================================
# =============================================================================
# Functions
# =============================================================================


def get_all_confidence(
    lddt_per_residue: torch.Tensor,
    ca_coordinates: torch.Tensor,
    ca_mask: torch.Tensor,
    cutoff=15.,
) -> float:
    """
    Compute an approximate LDDT score for the entire sequence

    lDDT reference:
    Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
    superposition-free score for comparing protein structures and models using
    distance difference tests. Bioinformatics 29, 2722–2728 (2013).

    Code below adopted from
    https://github.com/deepmind/alphafold/blob/1109480e6f38d71b3b265a4a25039e51e2343368/alphafold/model/lddt.py#L19

    Args:
        lddt_per_residue: the lddt score for each of the residues,
            of shape [num_res]
        ca_coordinates: the c-a coordinates of the residues,
            of shape [num_res, 3]
        ca_mask: mask of the c-a atoms,
            of shape [num_res]
        cutoff: The cutoff for each residue pair to be included

    Returns:
        The overall confidence for the entire prediction

    """

    assert ca_coordinates.ndim == 2
    assert lddt_per_residue.ndim == 1

    # Compute true and predicted distance matrices.
    dmat_true = torch.sqrt(
        torch.sum((ca_coordinates[:, None] - ca_coordinates[None, :])**2,
                  dim=-1).add(1e-10))

    dists_to_score = (
        torch.lt(dmat_true, cutoff) * ca_mask[..., :, None] *
        ca_mask[..., None, :] *
        (1. - torch.eye(dmat_true.shape[1], device=ca_mask.device))
        # Exclude self-interaction.
    )

    # Normalize over the appropriate axes.

    score = ((lddt_per_residue *
              (torch.sum(dists_to_score, dim=(-1, )).add(1e-10))).sum(-1) /
             (1e-10 + torch.sum(dists_to_score, dim=(-1, -2))))

    return score.item()


def _compute_confidence(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes per-residue pLDDT from logits.

    Code below adopted from
    https://github.com/deepmind/alphafold/blob/0be2b30b98f0da7aecb973bde04758fae67eb913/alphafold/common/confidence.py#L22

    Args:
        logits: the logits into the softmax, of shape [num_res, num_bins]

    Returns:
        predicted_lddt_ca: the predicted CA lddt, of shape [num_res]

    """
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bin_centers = torch.arange(start=0.5 * bin_width,
                               end=1.0,
                               step=bin_width,
                               device=logits.device)
    probs = torch.softmax(logits, dim=-1)
    confidence = torch.mv(probs, bin_centers)
    return confidence


# =============================================================================
# Classes
# =============================================================================


class ConfidenceHead(modules.OFModule):
    """
    This is the same pLDDT head from AF2, which provides a confidence measure
    of the model's prediction

    """

    def __init__(self, cfg: argparse.Namespace):
        super().__init__(cfg)
        self.network = nn.Sequential(
            nn.Linear(cfg.node_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.num_bins),
        )

    def forward(self, node_repr: torch.Tensor) -> torch.Tensor:
        node_repr = utils.normalize(node_repr)
        logits = self.network(node_repr)
        logits = _compute_confidence(logits)

        return logits


# =============================================================================
# Tests
# =============================================================================
if __name__ == '__main__':
    pass
