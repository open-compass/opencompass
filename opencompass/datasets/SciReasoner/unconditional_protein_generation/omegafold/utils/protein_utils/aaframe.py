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
This script contains the Frame object, that acts as an essential part to
convert to full atom coordinates for amino acids.
This is inspired by Jumper et al. (2021), where the authors refer to this
object as rigid group/affine update, and we unify the two notions here.

Some codes adopted from
https://github.com/deepmind/alphafold/blob/main/alphafold/model/all_atom.py
"""
# =============================================================================
# Imports
# =============================================================================
from typing import List, Tuple, Union

import torch
from torch.nn import functional as F

from ...utils.protein_utils import functions as f
from ...utils.protein_utils import residue_constants as rc

# =============================================================================
# Functions
# =============================================================================
# =============================================================================
# Constant
# =============================================================================
_BACKBONE_ROTATE = torch.tensor([
    [-1, 0., 0.],
    [0., 1., 0.],
    [0., 0., -1],
])


# =============================================================================
# Classes
# =============================================================================
class AAFrame(object):
    """
    The transformation object that holds translation and rotation
    """

    def __init__(self,
                 translation: torch.Tensor = None,
                 rotation: torch.Tensor = None,
                 mask: Union[torch.Tensor, torch.Tensor] = None,
                 safe: bool = True,
                 unit: str = 'Angstrom',
                 *,
                 expanded: bool = False) -> None:
        """
        Initialize the transformation

        Args:
            translation (): the translation vector of shape (*, 3)
            rotation (): the rotation vector of shape (*, 3, 3)
            mask (): the torsion_angles_mask tensor indicating the presence of
                the frame
            safe (): if to use safe initialization, if unsafe, it"s faster
            expanded (): if this frame is expanded to per-residue frames
        """
        super(AAFrame, self).__init__()
        self.orig = None
        if safe:
            self.mask = mask
            self.translation = translation
            self.rotation = rotation
        else:
            self._mask = mask
            self._translation = translation
            self._rotation = rotation

        self.expanded_ = expanded
        self._unit = unit

    @property
    def unit(self) -> str:
        """
        Get the unit of the frame

        Returns:
            the current unit of this frame

        """
        return self._unit

    def _assign(self, translation: torch.Tensor, rotation: torch.Tensor,
                unit: str, mask: torch.Tensor, in_place: bool,
                orig: str) -> 'AAFrame':
        """
        Create a new one or in-place assignment

        Args:
            translation: the translation (center) of the frame
            rotation: the rotation of the frame
            unit: the unit in which the frame operates
            mask: the mask of the frames indicating which components are valid
            in_place: if to perform the operation in-place
            orig: the info of the origin of the new frame

        Returns:
            A new frame, if not in-place, or the original frame with the
                attributes

        """
        if in_place:
            self._translation, self._rotation, = translation, rotation
            self._unit, self._mask = unit, mask
            return self
        else:
            return self._construct_frame(translation,
                                         rotation,
                                         mask,
                                         orig=orig,
                                         safe=True,
                                         unit=unit)

    def to_nanometers(self, in_place: bool = True) -> 'AAFrame':
        """
        Move the nanometers

        Args:
            in_place: if to perform the operation in place.

        Returns:

        """
        if self._unit == 'Angstrom':
            _translation = self._translation / 10
        else:
            _translation = self._translation
        _unit = 'nano'
        _rotation = self._rotation
        _mask = self._mask
        return self._assign(translation=_translation,
                            rotation=_rotation,
                            unit=_unit,
                            mask=_mask,
                            orig=f'To nano from {self}',
                            in_place=in_place)

    def to_angstrom(self, in_place: bool) -> 'AAFrame':
        """
        move to angstrom

        Args:
            in_place: if to use in_place operation

        Returns:

        """
        if self._unit == 'nano':
            _translation = self._translation * 10
        else:
            _translation = self._translation
        _unit = 'Angstrom'
        _rotation = self._rotation
        _mask = self._mask
        return self._assign(translation=_translation,
                            rotation=_rotation,
                            unit=_unit,
                            mask=_mask,
                            orig=f'To nano from {self}',
                            in_place=in_place)

    @property
    def translation(self) -> torch.Tensor:
        """
        Mask the ~self._translation by self.mask

        Returns:

        """
        return self._translation

    @translation.setter
    def translation(self, value: torch.Tensor) -> None:
        """
        Assign the translation in the frame with masked values set to 0"s.

        Args:
            value: the translation value

        """
        m = f.bit_wise_not(self.mask.unsqueeze(-1).expand_as(value))
        self._translation = value.masked_fill(m, 0)

    @property
    def rotation(self) -> torch.Tensor:
        """
        The rotation matrix

        Returns:

        """
        return self._rotation

    @rotation.setter
    def rotation(self, value: torch.Tensor) -> None:
        """
        Assign the rotation in the frame with masked values set to identity
        matrices.

        Args:
            value: the rotational matrices

        """
        mask = f.bit_wise_not(self.mask[..., None, None].expand_as(value))
        value = value.masked_fill(mask, 0.)
        value = value.masked_fill(
            mask * torch.eye(3, dtype=torch.bool).to(mask.device), 1)
        self._rotation = value

    @property
    def mask(self) -> torch.Tensor:
        """
        Hope this protects the attribute

        Returns:

        """
        return self._mask

    @mask.setter
    def mask(self, value: torch.Tensor):
        self._mask = value.bool()

    @classmethod
    def default_init(
        cls,
        *shape,
        unit: str = 'Angstrom',
        safe: bool = True,
        device: torch.device = torch.device('cpu'),
        mask: Union[torch.Tensor, torch.Tensor] = None,
    ) -> 'AAFrame':
        """
        partially initialize a bunch of frames, for now only supports one
        dimensional

        Args:
            shape (): the shape of the frames,
            mask (): the mask, if not provided, will be all true
            device (): on which will the frame reside
            safe (): if to safe init
            unit (): the unit

        Returns:

        """
        if mask is not None:
            assert tuple(mask.shape) == shape
        translation = torch.zeros(list(shape) + [3], device=device)
        rotation = torch.eye(3, dtype=translation.dtype,
                             device=device) * torch.ones(list(shape) + [1, 1],
                                                         device=device)
        if mask is None:
            mask = torch.ones_like(translation[..., 0], dtype=torch.bool)

        return cls._construct_frame(trans=translation,
                                    rots=rotation,
                                    mask=mask,
                                    orig='partially initialized',
                                    safe=safe,
                                    unit=unit)

    @classmethod
    def _neg_dim(cls, dim: int) -> Tuple[int, int, int]:
        if dim < 0:
            return dim, dim - 1, dim - 2
        else:
            return dim, dim, dim

    def unsqueeze(self, dim: int) -> 'AAFrame':
        """
        see torch.squeeze

        Args:
            dim ():

        Returns:

        """
        return self.dim_apply(torch.unsqueeze, dim=dim)

    def sum(self, dim: int, keepdim: bool = False) -> 'AAFrame':
        """
        see torch.sum

        Args:
            dim ():
            keepdim ():

        Returns:

        """
        dim0, dim1, dim2 = self._neg_dim(dim)
        m = torch.sum(self.mask, dim=dim0, keepdim=keepdim)
        t = torch.sum(self.translation, dim=dim1, keepdim=keepdim)
        r = torch.sum(self.rotation, dim=dim2, keepdim=keepdim)
        return self._construct_frame(t,
                                     r,
                                     m,
                                     f'Created by {torch.sum} at dim {dim}',
                                     safe=False,
                                     unit=self.unit)  # from self

    def dim_apply(self, func: callable, dim: int) -> 'AAFrame':
        """
        Apply torch functionals to the translation and rotations

        Args:
            func (): the functional to apply to
            dim (): the dimension to which the function will be applied

        Returns:

        """
        dim0, dim1, dim2 = self._neg_dim(dim)
        m = func(self.mask, dim0)
        t = func(self.translation, dim1)
        r = func(self.rotation, dim2)
        u = self.unit
        return self._construct_frame(t,
                                     r,
                                     m,
                                     f'Created by {func} at dim {dim}',
                                     safe=False,
                                     unit=u)  # from self

    @classmethod
    def _construct_frame(
        cls,
        trans: torch.Tensor,
        rots: torch.Tensor,
        mask: Union[torch.Tensor, torch.Tensor],
        orig: str,
        safe: bool,
        unit: str,
    ) -> 'AAFrame':
        """
        Construct a frame

        Args:
            trans: the absolute position in the bigger frame
            rots: the rotation of the frame
            mask: the mask indicating the validity of the frame
            orig: the message information about the origin of the frame
            unit: the unit for initialize
            safe: if use safe init

        Returns:

        """
        # assert t.shape[:-1] == r.shape[:-2] == m.shape
        transformation = AAFrame(translation=trans,
                                 rotation=rots,
                                 mask=mask,
                                 safe=safe,
                                 unit=unit)
        transformation.orig = orig

        return transformation

    @classmethod
    def from_4x4(cls, m: torch.Tensor, mask: torch.Tensor,
                 unit: str) -> 'AAFrame':
        """
        get the frames from 4x4 matrix

        Args:
            m (): the transformation in homogeneous coordinates
                should be of shape (*, 4, 4)
            mask (): the masking tensor
            unit ():

        Returns:
            A transformation

        """

        return cls._construct_frame(m[..., 0:3, 3],
                                    m[..., 0:3, 0:3],
                                    mask=mask,
                                    orig='from matrix',
                                    safe=True,
                                    unit=unit)

    def transform(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Apply the transformation on the input coordinates

        Args:
            pos (): the 3-D coordinates to transforms,
                 of shape (*, 3)

        Note:
            if we are using batched dims, we simply assume that the
            dimensions of pos can be split into three parts
            1. the batched_dims
            2. the ones to do the outer-product-like expansion
            3. the 3 xyz coordinate value

        Returns:
            transformed coordinates of the same shape as the coordinates,
                of shape (N, 3)

        Examples:
            >>> frames = AAFrame(
            ...     translation=torch.zeros(10,3),
            ...     rotation=torch.eye(3)[None, ...].repeat(10, 1, 1),
            ...     mask=torch.ones(10, dtype=torch.bool)
            ... )
            >>> frames.shape
            torch.Size([10])
            >>> frames.transform(torch.randn(10, 3)).shape
            torch.Size([10, 3]) # one-to-one
            >>> frames.transform(torch.randn(10, 1, 3)).shape
            torch.Size([10, 1, 3])  # it is still one-to-one
            >>> frames.transform(torch.randn(1, 4, 3)).shape
            torch.Size([10, 4, 3])  # this broadcasts to every pair,
                                    # with the first dimension being the
                                    # frames
            >>> frames.transform(torch.randn(4, 1, 3)).shape
            torch.Size([10, 1, 3])
            >>> frames = AAFrame(
            ...     translation=torch.zeros(10, 9, 3),
            ...     rotation=torch.eye(3)[None, ...].repeat(10, 9, 1, 1),
            ...     mask=torch.ones(10, 9, dtype=torch.bool)
            ... )
            >>> frames.shape
            torch.Size([10, 9])
            >>> frames.transform(torch.randn(10, 9, 3)).shape
            torch.Size([10, 9, 3])
            >>> frames.transform(torch.randn(10, 1, 3)).shape
            torch.Size([10, 9, 3])  # this broadcasts to 9, but does not
                                    # work with shape (1, 9, 3)
            >>> frames.transform(torch.randn(1, 1, 3)).shape
            torch.Size([10, 9, 3])  #
            >>> frames.transform(torch.randn(10, 9, 4, 3)).shape
            torch.Size([10, 9, 4, 3])
            >>> frames.transform(torch.randn(10, 1, 9, 4, 3)).shape
            torch.Size([10, 9, 9, 4, 3])    # the 1st, 2nd dim are from frames
            >>> frames.transform(torch.randn(10, 1, 1, 3)).shape
            torch.Size([10, 9, 1, 3])
        """
        batched_dims = len(self.shape)
        shape1 = self.shape[:batched_dims]
        shape2 = pos.shape[batched_dims:-1]  # the ones to cross
        self_shape2 = self.shape[batched_dims:]
        out = self.view(*shape1, *[1 for _ in range(len(shape2))],
                        *self_shape2)
        return f.batch_matrix_vector(out.rotation, pos) + out.translation

    @classmethod
    def from_torsion(
        cls,
        unit: str,
        torsion_angles: torch.Tensor,
        mask: Union[torch.Tensor, torch.Tensor],
        translation: torch.Tensor = None,
    ) -> 'AAFrame':
        """
        Create a transformation that rotates around the x-axis

        Args:
            unit ():
            torsion_angles (): the torsion angle to create the axis with,
                should be of shape (*, 2)
            mask (): the masking tensor
            translation (): optional, if provided will be passed in to the
                transformation

        Returns:
            A rotation matrix around the x axis

        """
        device = torsion_angles.device
        _make_rot_mat = torch.tensor(
            [
                [0., 0., 0., 0., 0., -1, 0., 1., 0.],  # sin
                [0., 0., 0., 0., 1., 0., 0., 0., 1.],  # cos
            ],
            dtype=torsion_angles.dtype,
            device=device)
        rot_mat = torch.matmul(torsion_angles, _make_rot_mat)

        rot_mat = rot_mat.unflatten(dim=-1, sizes=[3, 3])
        rot_mat[..., 0, 0] = 1

        if translation is None:
            shape = list(torsion_angles.shape)
            shape[-1] = 3
            translation = torch.zeros(*shape, device=device)

        return cls._construct_frame(translation,
                                    rot_mat,
                                    mask,
                                    'from torsion',
                                    safe=True,
                                    unit=unit)

    def __getitem__(self, idx: Union[slice, int, torch.Tensor]) -> 'AAFrame':
        """
        Select the frame

        Args:
            idx (): the index of the selection

        Returns:
            selected transformation

        """
        if isinstance(idx, (slice, int)):
            return self._construct_frame(self.translation[..., idx, :],
                                         self.rotation[..., idx, :, :],
                                         self.mask[..., idx],
                                         f'selected from {self} at {idx}',
                                         unit=self.unit,
                                         safe=False)
        elif isinstance(idx, torch.Tensor):
            return self._construct_frame(self.translation[idx, :],
                                         self.rotation[idx, :, :],
                                         self.mask[idx],
                                         f'selected from {self} by tensor',
                                         unit=self.unit,
                                         safe=False)
        else:
            raise IndexError(f'Type {type(idx)} not supported for indexing')

    def __setitem__(self, key: Union[int, torch.Tensor, List[int]],
                    value: Union[torch.Tensor, 'AAFrame']) -> None:
        if isinstance(value, AAFrame):
            t = value.translation.to(self._translation.dtype)
            r = value.rotation.to(self._rotation.dtype)
            m = value.mask.to(self._mask.dtype)
        else:
            t = r = value
            m = bool(value)
        mask = self.mask.clone()
        translation = self.translation.clone()
        rotation = self.rotation.clone()

        if isinstance(key, int):
            mask[..., key] = m
            translation[..., key, :] = t
            rotation[..., key, :, :] = r
        elif isinstance(key, (torch.Tensor, list)):
            # this because it cannot use in-place operations for gradients
            mask[key] = m
            translation[key, :] = t
            rotation[key, :, :] = r

        self.mask = mask
        self.translation = translation
        self.rotation = rotation

    @property
    def device(self) -> torch.device:
        """

        Returns:

        """
        assert (self._mask.device == self._translation.device ==
                self._rotation.device)
        return self._mask.device

    @property
    def shape(self) -> torch.Size:
        """

        Returns: the shape of the tensor

        """
        return self.mask.shape

    def __mul__(self, other) -> 'AAFrame':
        if isinstance(other, AAFrame):
            return self._combine_transformation(other)
        else:
            return self._tensor_multiplication(other)

    def _tensor_multiplication(self, other: torch.Tensor) -> 'AAFrame':
        """
        Multiply everything by the tensor

        Args:
            other:

        Returns:

        """
        if torch.logical_or(torch.eq(other, 0), torch.eq(other, 1)).all():
            m = self.mask * other
            t = self.translation * other[..., None]
            r = self.rotation * other[..., None, None]
        else:
            t = self.translation * other
            m = self.mask
            r = self.rotation

        return self._construct_frame(t,
                                     r,
                                     m,
                                     f'Created by multiplication from {self}',
                                     safe=False,
                                     unit=self.unit)

    def _combine_transformation(self, other: 'AAFrame') -> 'AAFrame':
        """
        Combine two frames

        Args:
            The following two arguments all have the transition of shape
            (N, 3) at the first place and rotation matrix of shape (N, 3, 3) at
             the second

            other (): frame 1

        Returns:
            the end frame

        """
        # the rotation
        if self.shape != other.shape:
            t_1 = self.translation[..., None, :].expand_as(other.translation)
            r_1 = self.rotation[..., None, :, :].expand_as(other.rotation)
            m_1 = self.mask[..., None].expand_as(other.mask).reshape(-1)
            t_1, r_1 = t_1.reshape(-1, 3), r_1.reshape(-1, 3, 3)
        else:
            t_1, r_1, m_1 = self.translation, self.rotation, self.mask
            t_1, r_1, m_1 = t_1.view(-1, 3), r_1.view(-1, 3, 3), m_1.view(-1)

        if self.unit == 'Angstrom':
            other.to_angstrom(in_place=True)
        else:
            other.to_nanometers(in_place=True)
        t_2, r_2 = other.translation.view(-1, 3), other.rotation.view(-1, 3, 3)
        m_2 = other.mask.view(-1)

        r_out = torch.bmm(r_1, r_2)
        # the transition
        t_out = t_1 + f.batch_matrix_vector(r_1, t_2)
        # the torsion_angles_mask
        m_out = m_1 * m_2

        return self._construct_frame(t_out.view(*other.shape, 3),
                                     r_out.view(*other.shape, 3, 3),
                                     m_out.view(*other.shape),
                                     f'Combination of {self} and {other}',
                                     safe=False,
                                     unit=self.unit)

    def __repr__(self) -> str:
        return f'Frame {id(self)}'

    def view(self, *args) -> 'AAFrame':
        """
        See Tensor.view

        Args:
            *args ():

        Returns:

        """
        mask = self.mask
        translation = self.translation
        rotation = self.rotation
        return self._construct_frame(translation.view(*args, 3),
                                     rotation.view(*args, 3, 3),
                                     mask.view(*args),
                                     f'view from {self}',
                                     safe=False,
                                     unit=self.unit)

    @property
    def dtype(self):
        return self.translation.dtype

    def expand_w_torsion(self, torsion_angles: torch.Tensor,
                         torsion_angles_mask: torch.Tensor,
                         fasta: torch.Tensor) -> 'AAFrame':
        r"""
        Compute the global frame

        Lines 2-10
        Algorithm 24, Page 31 of the AlphaFold 2 supplementary material

        Args:
            self (): the transformation from backbone to global
                bb_coor (): the transition of the backbone transformation, or
                the coordinates of the CA atom,
                    should be of shape (N, 3)
                bb_rot (): the rotation of the backbone transformation,
                    should be of shape (N, 3, 3)
            torsion_angles (): the torsion angles
                (\omega, \phi, \psi, \chi_1, \chi_2, \chi_3, \chi_4)
                should be of shape (N, 7, 2)
            torsion_angles_mask (): the torsion angle masks indicating presence
                (\omega, \phi, \psi, \chi_1, \chi_2, \chi_3, \chi_4)
                should be of shape (N, 7)
            fasta (): input sequence where each place is an index indicating
                which amino acid is in each position, following ~restypes

        Returns:
            Frame

        """
        assert self.unit == 'Angstrom'
        if torsion_angles.shape[-2] == 5:
            torsion_angles = torch.cat((torch.zeros_like(
                torsion_angles[..., 0:2, :]), torsion_angles),
                                       dim=-2)
            torsion_angles_mask = torch.cat((torch.zeros_like(
                torsion_angles_mask[..., 0:2]), torsion_angles_mask),
                                            dim=-1)

        # append an identity for backbone2backbone
        shape = list(torsion_angles.shape)
        shape[-2] = 1
        angle = torch.tensor([[0, 1]], dtype=self.dtype,
                             device=self.device).expand(shape)  # (*, 1, 2)
        angle_mask = torch.tensor([True], dtype=torch.bool,
                                  device=self.device).expand(shape[:-1])
        torsion_angles = torch.cat((angle, torsion_angles), -2)  # (*, 8, 2)
        torsion_angles_mask = torch.cat((angle_mask, torsion_angles_mask), -1)

        # prepare the angles
        torsion_angles = f.robust_normalize(torsion_angles)
        rot_x = AAFrame.from_torsion(torsion_angles=torsion_angles,
                                     mask=torsion_angles_mask,
                                     unit='Angstrom')

        # make extra backbone frames
        # This follows the order of ~restypes
        m = rc.restype_aa_default_frame.to(self.device)[fasta]
        default_frames = AAFrame.from_4x4(m,
                                          torsion_angles_mask,
                                          unit='Angstrom')
        all_frames = default_frames * rot_x
        # make side chain frames (chain them up along the side chain)
        chi2_frame_to_frame = all_frames[5]
        chi3_frame_to_frame = all_frames[6]
        chi4_frame_to_frame = all_frames[7]
        # chains
        chi1_frame_to_backb = all_frames[4]
        chi2_frame_to_backb = chi1_frame_to_backb * chi2_frame_to_frame
        chi3_frame_to_backb = chi2_frame_to_backb * chi3_frame_to_frame
        chi4_frame_to_backb = chi3_frame_to_backb * chi4_frame_to_frame

        # all_frames[4] = chi1_f2bb
        all_frames[5] = chi2_frame_to_backb
        all_frames[6] = chi3_frame_to_backb
        all_frames[7] = chi4_frame_to_backb
        # get all
        # map atom literature positions to the global frame
        all_f2global = self * all_frames
        all_f2global.expanded_ = True

        return all_f2global

    def rotate(self, rotation: torch.Tensor):
        """
        Rotate with a rotation matrix

        Note:
            batched rotated not yet supported,
            for now just use ~Frame._construct_transformation

        Args:
            rotation (): the rotation matrix of shape (d, d)

        Returns:
            Rotated frame

        """
        if len(rotation.shape) == 2:
            t = self.translation
            r = torch.matmul(self.rotation, rotation)
            return self._construct_frame(t,
                                         r,
                                         self.mask,
                                         f'Rotated from {self}',
                                         safe=False,
                                         unit=self.unit)
        else:
            raise NotImplementedError('Not yet implemented')

    def expanded_to_pos(
            self,
            fasta: torch.Tensor,
            full: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the full atom representation

        Args:
            fasta: the sequence to compute the atoms
            full: if to use safe initialization

        Returns:
            the atom14 representation and the mask indicating the presence
            of the atoms

        """
        if full:
            assert self.expanded_
            num_classes = 8
            frame = self
            pos_counts = 14
        else:
            num_classes = 1
            frame = self.unsqueeze(-1)
            pos_counts = 5

        assert self._unit == 'Angstrom'

        fasta = fasta.cpu()
        residx2group = rc.restype_atom14_to_aa
        residx2group = residx2group[..., :pos_counts]
        residx2group = residx2group[fasta].to(self.device)
        group_mask = F.one_hot(residx2group, num_classes=8)
        group_mask = group_mask[..., :num_classes]
        group_mask = group_mask * frame.mask[..., None, :]
        to_mask = frame.unsqueeze(-2) * group_mask
        map_atoms_to_global = to_mask.sum(-1)
        lit_pos = rc.restype_atom14_aa_positions
        lit_pos = lit_pos[..., :pos_counts, :]
        lit_pos = lit_pos[fasta].to(self.device)
        pred_pos = map_atoms_to_global.transform(lit_pos)
        # mask = c.restype_atom14_mask[sequence]  # (N, 14)
        # mask |= self.mask[..., None]
        pred_pos = pred_pos * map_atoms_to_global.mask[..., None]

        return pred_pos, torsion_mask_to_atom14_mask(frame.mask,
                                                     group_mask,
                                                     fasta=fasta)

    def __len__(self):
        return len(self.mask)

    @property
    def inverse(self) -> 'AAFrame':
        """
        The inverse of the transformation

        Returns:

        """
        r = self.rotation.transpose(-1, -2)
        t = f.batch_matrix_vector(r, self.translation)
        return self._construct_frame(-t,
                                     r,
                                     self.mask,
                                     f'inversed from {self}',
                                     safe=False,
                                     unit=self.unit)

    def position_in_frame(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Get the frame-based position of the given global position

        Args:
            pos (): the global position of shape (*, 3)

        Returns:
            the result

        """
        return self.inverse.transform(pos)

    @classmethod
    def from_tensor(cls, tensor, unit: str) -> 'AAFrame':
        """
        Args:
            tensor: (*, 7)
            unit:
        """
        q_dim = 4 if tensor.shape[-1] == 7 else 3
        quaternion, tx, ty, tz = torch.split(tensor, [q_dim, 1, 1, 1], dim=-1)
        rotation = f.quaternion_to_matrix(quaternion)
        translation = torch.stack([tx[..., 0], ty[..., 0], tz[..., 0]], dim=-1)

        return cls._construct_frame(trans=translation,
                                    rots=rotation,
                                    mask=torch.ones_like(translation[..., 0]),
                                    orig='from tensor',
                                    safe=True,
                                    unit=unit)


def torsion_mask_to_atom14_mask(torsion_mask: torch.Tensor,
                                group_mask: torch.Tensor,
                                fasta: torch.Tensor) -> torch.Tensor:
    """
    expand the mask of torsion angles into atom14 masks

    Args:
        torsion_mask (): the mask for torsion angles, of shape (*, 8)
        group_mask (): the group mask to add on, of shape (*, 14, 8)
        fasta (): the sequence for this operation

    Returns:
        Expanded mask of shape (*, 14)

    """
    atom14_exist_mask = group_mask[..., 1:].sum(-1)
    atom14_exist_mask[..., 4] = fasta != 7
    atom14_exist_mask[..., 0:3] = torsion_mask[..., 0:1]
    return atom14_exist_mask.bool()


# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Tests
# =============================================================================
if __name__ == '__main__':
    pass
