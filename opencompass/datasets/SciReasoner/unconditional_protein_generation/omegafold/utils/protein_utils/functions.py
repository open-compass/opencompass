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
This script contains some functions that may be handy somewhere
"""
# =============================================================================
# Constants
# =============================================================================
# =============================================================================
# Imports
# =============================================================================
import typing

import torch


# =============================================================================
# Functions
# =============================================================================
def get_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Replacement for LA.norm since MPS does not support it yet.

    Args:
        x:

    Returns:

    """
    return x.norm(p=2, dim=-1)


def robust_normalize(x: torch.Tensor,
                     dim: int = -1,
                     p: typing.Union[int, str] = 2) -> torch.Tensor:
    """
    Normalization with a constant small term on the denominator

    Args:
        x (): tensor to normalize
        dim (): the dimension along which to perform the normalization
        p (): the p in l-p

    Returns:
        the normalized result

    """
    return x / (x.norm(p=p, dim=dim, keepdim=True).clamp(4e-5))


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    # The following from PyTorch3d
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4) or (..., 3).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if quaternions.shape[-1] == 3:
        quaternions = torch.cat(
            (torch.ones_like(quaternions[..., 0:1]), quaternions), dim=-1)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def batch_matrix_vector(matrix: torch.Tensor,
                        vector: torch.Tensor) -> torch.Tensor:
    """
    Perform batched matrix vector product on the last dimension

    Args:
        matrix (): of shape (*, d, d)
        vector (): of shape (*, d)

    Returns:
        the product of the two

    """
    assert len(matrix.shape[:-2]) == len(vector.shape[:-1])

    return torch.einsum('...cd, ...d -> ...c', matrix, vector)


def create_pseudo_beta(atom_pos: torch.Tensor,
                       atom_mask: torch.Tensor) -> torch.Tensor:
    """

    Args:
        atom_pos: the atom position in atom14 format,
            of shape [*, num_res, 14, 3]
        atom_mask: the atom mask in atom14 format,
            of shape [*, num_res, 14]

    Returns:
        CB coordinate (when available) and CA coordinate (when not available)

    """
    if not (atom_mask.shape[-1] == atom_pos.shape[-2] == 14):
        raise ValueError('Only supports atom 14')
    pseudo_beta = torch.where(
        atom_mask[..., 4:5].expand(list(atom_mask.shape[:-1]) + [3]).bool(),
        atom_pos[..., 4, :], atom_pos[..., 1, :])
    return pseudo_beta


def bit_wise_not(boolean_tensor: torch.Tensor) -> torch.Tensor:
    """For MPS devices that have no support for yet bit-wise not"""
    boolean_tensor = 1 - boolean_tensor.float()
    return boolean_tensor.bool()


# =============================================================================
# Tests
# =============================================================================
if __name__ == '__main__':
    pass
