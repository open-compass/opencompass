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

from . import modules, utils
from .utils import residue_constants as rc


# =============================================================================
# Constants
# =============================================================================
# =============================================================================
# Functions
# =============================================================================
def _get_pos(shape: torch.Size, device: torch.device, dtype: torch.dtype,
             seq_dim: typing.Tuple[int, ...]) -> torch.Tensor:
    """Get the position of the tokens given

    Args:
        shape: the shape of the tensor to be applied with RoPE
        device: the device on which the tensor reside
        dtype: the datatype of the tensor
        seq_dim: dimensions of the tensor that reference the sequence length

    Returns:
        The position tensor of the shape from ~shape indexed by seq_dim

    """
    spatial_shape = [shape[i] for i in seq_dim]
    total_len = 1
    for i in spatial_shape:
        total_len *= i
    position = torch.arange(total_len, dtype=dtype, device=device)
    position = position.reshape(*spatial_shape)

    return position


def _apply_embed(inputs: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor,
                 seq_dim: typing.Tuple[int, ...]) -> torch.Tensor:
    """Applies RoPE to ~inputs

    Args:
        inputs: the tensor to which RoPE is applied, the dimensions indexed by
            ~seq_dim indicates the spatial dimensions
        sin: the sine tensor that constitutes parts of the RoPE,
            of spatial shape + vector dimension
        cos: the cosine tensor that constitutes parts of the RoPE,
            of spatial shape + vector dimension
        seq_dim: the dimensions indicating the spatial dimensions,
            must be consecutive

    Returns:
        tensor with RoPE applied.

    """
    gaps = [(seq_dim[i + 1] - seq_dim[i]) == 1
            for i in range(len(seq_dim) - 1)]
    if len(gaps) > 0:
        if not all(gaps):
            raise ValueError(f'seq_dim must be consecutive, but got {seq_dim}')

    # Align dimensions of sine and cosine
    seq_dim = sorted(seq_dim)
    end = seq_dim[-1]
    for _ in range(seq_dim[0]):
        sin = sin.unsqueeze(0)
        cos = cos.unsqueeze(0)
        end += 1

    for _ in range(end, inputs.ndim - 1):
        sin = sin.unsqueeze(_)
        cos = cos.unsqueeze(_)

    # Apply RoPE
    x1, x2 = torch.split(inputs, inputs.shape[-1] // 2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# =============================================================================
# Classes
# =============================================================================
class EdgeEmbedder(modules.OFModule):
    """
    Embed the input into node and edge representations

    """

    def __init__(self, cfg: argparse.Namespace) -> None:
        super(EdgeEmbedder, self).__init__(cfg)

        self.proj_i = nn.Embedding(cfg.alphabet_size, cfg.edge_dim)
        self.proj_j = nn.Embedding(cfg.alphabet_size, cfg.edge_dim)
        self.relpos = RelPosEmbedder(cfg.relpos_len * 2 + 1, cfg.edge_dim)

    def forward(self, fasta_sequence: torch.Tensor,
                out: torch.Tensor) -> torch.Tensor:
        out += self.proj_i(fasta_sequence).unsqueeze(-2)
        out += self.proj_j(fasta_sequence).unsqueeze(-3)
        out += self.relpos(fasta_sequence.size(-1))

        return out


class RoPE(nn.Module):
    """The RoPE module

    Attributes:
        input_dim: the dimension of the input vectors.

    """

    def __init__(self, input_dim: int) -> None:
        super(RoPE, self).__init__()
        if input_dim % 2 != 0:
            raise ValueError(
                f'Input dimension for RoPE must be a multiple of 2,'
                f' but got {input_dim}')
        self.input_dim = input_dim
        self.half_size = input_dim // 2
        freq_seq = torch.arange(self.half_size, dtype=torch.float32)
        freq_seq = -freq_seq.div(float(self.half_size))

        self.register_buffer('inv_freq',
                             torch.pow(10000., freq_seq),
                             persistent=False)

    def forward(self, tensor: torch.Tensor,
                seq_dim: typing.Union[int, tuple]) -> torch.Tensor:
        """

        Args:
            tensor: the tensor to apply rope onto
            seq_dim: the dimension that represents the sequence dimension

        Returns:

        """
        if isinstance(seq_dim, int):
            seq_dim = [
                seq_dim,
            ]
        sin, cos = self._compute_sin_cos(tensor, seq_dim)

        return _apply_embed(tensor, sin, cos, seq_dim)

    def _compute_sin_cos(
        self, tensor: torch.Tensor, seq_dim: typing.Tuple[int]
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Compute sine and cosine tensors

        Args:
            tensor: the tensors to apply RoPE to
            seq_dim: the dimension indices of the spatial dimensions

        Returns:
            A tuple of tensors where the first one is the sine tensor
                and the second one is the cosine tensor

        """
        position = _get_pos(tensor.shape, tensor.device, tensor.dtype, seq_dim)
        sinusoid = torch.einsum('..., d->...d', position, self.inv_freq)
        sin, cos = torch.sin(sinusoid), torch.cos(sinusoid)
        return sin, cos


class RelPosEmbedder(nn.Embedding):
    """
        Compute the relative positional embedding,
        this is the same algorithm in
        Jumper et al. (2021) Suppl. Alg. 4 "relpos"
    """

    def forward(self, num_res: int) -> torch.Tensor:
        """

        Args:
            num_res: number of residues in input sequence.

        Returns:

        """
        idx = torch.arange(num_res, device=next(self.parameters()).device)
        one_side = self.num_embeddings // 2
        idx = (idx[None, :] - idx[:, None]).clamp(-one_side, one_side)
        idx = idx + one_side
        return super(RelPosEmbedder, self).forward(idx)  # [num_res, dim]


class StructEmbedder(modules.OFModule):
    """
    Encoder for pair wise atom distance without distance clamp
    but a sublinear-function with ord encoder.
    """

    def __init__(self, cfg: argparse.Namespace):
        super(StructEmbedder, self).__init__(cfg)
        self.rough_dist_bin = modules.Val2ContBins(cfg.rough_dist_bin)
        self.dist_bin = modules.Val2ContBins(cfg.dist_bin)
        self.pos_bin = modules.Val2ContBins(cfg.pos_bin)

        self.aa_embedding = nn.Embedding(21 * 21, embedding_dim=cfg.c)

        frame_num = 8
        atom_num = 14

        self.dist_bin_embedding = nn.Linear(cfg.dist_bin.x_bins, cfg.c)
        self.rough_dist_bin_embedding = nn.Linear(cfg.rough_dist_bin.x_bins,
                                                  cfg.c)

        self.dist_bin_linear = nn.Linear(atom_num * atom_num * cfg.c, cfg.c)
        self.rough_dist_bin_linear = nn.Linear(atom_num * atom_num * cfg.c,
                                               cfg.c)

        self.pos_bin_embedding = nn.Linear(cfg.pos_bin.x_bins, cfg.c)
        self.pos_linear = nn.Linear(frame_num * atom_num * 3 * cfg.c, cfg.c)

        self.linear_z_weights = nn.Parameter(
            torch.zeros([cfg.c, cfg.c, cfg.edge_dim]))
        self.linear_z_bias = nn.Parameter(torch.zeros([cfg.edge_dim]))

    def forward(
        self,
        fasta1: torch.Tensor,
        fasta2: torch.Tensor,
        pos14_a: torch.Tensor,
        mask14_a: torch.Tensor,
        pos14_b: torch.Tensor,
        mask14_b: torch.Tensor,
        frame8: utils.AAFrame,
    ):
        pairwise_fasta = fasta1.unsqueeze(-1) * 21 + fasta2.unsqueeze(-2)
        d = torch.norm(pos14_b[None, :, None] - pos14_a[:, None, :, None],
                       p=2,
                       dim=-1,
                       keepdim=False)
        d_mask = mask14_b[None, :, None] * mask14_a[:, None, :, None]
        d_mask = d_mask.unsqueeze(-1)
        local_mask = torch.mul(mask14_b[None, :, None], frame8.mask[:, None, :,
                                                                    None])
        local_mask = local_mask.unsqueeze(-1)

        local_vec = frame8.unsqueeze(1).unsqueeze(-1).position_in_frame(
            pos14_b[None, :, None, :])

        return self._sharded_compute(pairwise_fasta, d, local_vec, d_mask,
                                     local_mask)

    def _sharded_compute(self, pairwise_fasta: torch.Tensor, d: torch.Tensor,
                         local_vec: torch.Tensor, d_mask: torch.Tensor,
                         local_mask: torch.Tensor) -> torch.Tensor:
        pairwise_fasta = self.aa_embedding(pairwise_fasta)
        d1 = self.rough_dist_bin(d)
        d2 = self.dist_bin(d)
        d3 = self.pos_bin(local_vec)

        d1 = self.rough_dist_bin_embedding(d1)
        d1 = d1 * d_mask
        d1 = self.rough_dist_bin_linear(d1.flatten(start_dim=-3))

        d2 = self.dist_bin_embedding(d2)
        d2 = d2 * d_mask
        d2 = self.dist_bin_linear(d2.flatten(start_dim=-3))

        d3 = self.pos_bin_embedding(d3)
        d3 = d3 * (local_mask.unsqueeze(-1))
        d3 = self.pos_linear(d3.flatten(start_dim=-4))

        final_d = d1 + d2 + d3  # + d4
        O_ = torch.einsum('...sdi,...sdj->...sdij', pairwise_fasta, final_d)
        Z = torch.einsum('...sdij,ijh->...sdh', O_,
                         self.linear_z_weights) + self.linear_z_bias
        return Z


class PairStructEmbedder(StructEmbedder):

    def forward(
        self,
        fasta: torch.Tensor,
        pos14: torch.Tensor,
        pos14_mask: torch.Tensor,
        frame8: utils.AAFrame,
    ):
        return super(PairStructEmbedder, self).forward(fasta1=fasta,
                                                       fasta2=fasta,
                                                       pos14_a=pos14,
                                                       pos14_b=pos14,
                                                       mask14_a=pos14_mask,
                                                       mask14_b=pos14_mask,
                                                       frame8=frame8)


class RecycleEmbedder(modules.OFModule):
    """
    The recycle embedder from Jumper et al. (2021)

    """

    def __init__(self, cfg: argparse.Namespace):
        super(RecycleEmbedder, self).__init__(cfg)

        self.layernorm_node = nn.LayerNorm(cfg.node_dim)
        self.layernorm_edge = nn.LayerNorm(cfg.edge_dim)
        self.dgram = modules.Val2Bins(cfg.prev_pos)
        self.prev_pos_embed = nn.Embedding(
            cfg.prev_pos.num_bins,
            cfg.edge_dim,
        )
        if cfg.struct_embedder:
            self.embed_struct = PairStructEmbedder(cfg)

    def forward(
        self,
        fasta: torch.Tensor,
        prev_node: torch.Tensor,
        prev_edge: torch.Tensor,
        prev_x: torch.Tensor,
        node_repr: torch.Tensor,
        edge_repr: torch.Tensor,
        atom14_mask: torch.Tensor,
        prev_frames: utils.AAFrame,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Recycle the last run

        Args:
            fasta:
            prev_node: node representations from the previous cycle
                of shape [num_res, node_repr_dim]
            prev_edge: edge representations from the previous cycle
                of shape [num_res, num_res, edge_repr_dim]
            prev_x: pseudo beta coordinates from the previous cycle.
                of shape [num_res, 3]
            node_repr: the node representation to put stuff in
            edge_repr: the edge representation to put stuff in
            atom14_mask: the mask for the 14 atoms
            prev_frames: the frames from the previous cycle

        Returns:

        """
        atom_mask = rc.restype2atom_mask.to(self.device)[fasta]
        prev_beta = utils.create_pseudo_beta(prev_x, atom_mask)
        d = utils.get_norm(prev_beta.unsqueeze(-2) - prev_beta.unsqueeze(-3))
        d = self.dgram(d)
        node_repr[..., 0, :, :] = (node_repr[..., 0, :, :] +
                                   self.layernorm_node(prev_node))
        edge_repr += self.prev_pos_embed(d)
        edge_repr += self.layernorm_edge(prev_edge)
        if self.cfg.struct_embedder:
            edge_repr += self.embed_struct(fasta, prev_x, atom14_mask,
                                           prev_frames)

        return node_repr, edge_repr


# =============================================================================
# Tests
# =============================================================================
if __name__ == '__main__':
    pass
