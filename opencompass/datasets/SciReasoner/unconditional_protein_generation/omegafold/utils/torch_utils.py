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
PyTorch utilities
"""
# =============================================================================
# Imports
# =============================================================================
import numbers
import typing

import torch
from torch.nn import functional as F

# =============================================================================
# Constants
# =============================================================================

T = typing.TypeVar('T')


# =============================================================================
# Functions
# =============================================================================
def mask2bias(mask: torch.Tensor, *, inf: float = 1e9) -> torch.Tensor:
    """Convert mask to attention bias

    Args:
        mask: the mask to convert to bias representation
        inf: the floating point number to represent infinity

    Returns:
        bias representation for masking in attention

    """
    return mask.float().sub(1).mul(inf)


def normalize(inputs: torch.Tensor,
              normalized_shape: typing.Optional[typing.Union[
                  int, typing.List[int], torch.Size]] = None,
              in_place: bool = False) -> torch.Tensor:
    """Layer normalization without a module (and weight)

    Args:
        inputs: the input tensor to be normalized
        normalized_shape: the normalized_shape for normalization
        in_place: if to perform the operations in-place

    Returns:
        normalized tensor

    """
    if normalized_shape is None:
        normalized_shape = inputs.shape[-1]
    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = (normalized_shape, )

    if in_place:
        # This seems to create small discrepancy in result
        dim = list(range(len(inputs.shape))[-len(normalized_shape):])
        inputs -= inputs.mean(dim=dim, keepdim=True)
        inputs *= torch.rsqrt(inputs.var(dim=dim, keepdim=True) + 1e-5)
        return inputs
    else:
        # F.layer_norm seems a bit faster
        return F.layer_norm(inputs, normalized_shape, None, None, 1e-5)


def masked_mean(values: torch.Tensor,
                mask: torch.Tensor,
                dim: typing.Union[int, typing.Sequence[int], None],
                keepdim: typing.Optional[bool] = False,
                eps: typing.Optional[float] = 4e-5) -> torch.Tensor:
    """Mean operation with mask

    Args:
        values: the values to take the mean for
        mask: the mask to take the mean with
        dim: the dimension along which to take the mean
        keepdim: to keep the dimension
        eps: the epsilon to compute mean for

    Returns:
        mean result

    """
    values = values.masked_fill(~mask.bool(), 0).sum(dim, keepdim=keepdim)
    norm = mask.sum(dim, keepdim=keepdim, dtype=values.dtype) + eps
    return values / norm


def recursive_to(obj: typing.Any, **kwargs) -> typing.Any:
    r"""
    Just to move things to space
    *args is removed because it brings problems in using .cpu()

    Args:
        obj (): the object to move
        kwargs (): different keyword arguments

    Returns:
        cuda tensors in its original construct

    """
    if isinstance(obj, torch.Tensor):
        try:
            return obj.to(**kwargs)
        except RuntimeError:
            kwargs.pop('non_blocking')
            return obj.to(**kwargs)
    elif isinstance(obj, list):
        return [recursive_to(o, **kwargs) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, **kwargs) for o in obj)
    elif isinstance(obj, set):
        return set(recursive_to(o, **kwargs) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, **kwargs) for k, v in obj.items()}
    elif hasattr(obj, 'to'):
        # this takes care of classes that implements the ~to method
        return obj.to(**kwargs)
    else:
        return obj


# =============================================================================
# Classes
# =============================================================================
# =============================================================================
# Tests
# =============================================================================
if __name__ == '__main__':
    pass
