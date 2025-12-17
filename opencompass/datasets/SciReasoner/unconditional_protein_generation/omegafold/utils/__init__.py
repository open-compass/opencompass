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
# flake8: noqa
"""

"""
# =============================================================================
# Imports
# =============================================================================
from typing import Dict, Union  # noqa: F401, F403

import torch  # noqa: F401, F403

from ..utils.protein_utils import residue_constants  # noqa: F401, F403
from ..utils.protein_utils.aaframe import AAFrame  # noqa: F401, F403
from ..utils.protein_utils.functions import bit_wise_not  # noqa: F401, F403
from ..utils.protein_utils.functions import \
    robust_normalize  # noqa: F401, F403
from ..utils.protein_utils.functions import create_pseudo_beta, get_norm
from ..utils.torch_utils import masked_mean  # noqa: F401, F403
from ..utils.torch_utils import normalize  # noqa: F401, F403
from ..utils.torch_utils import mask2bias, recursive_to

# =============================================================================
# Constants
# =============================================================================
DATA = Dict[str, Union[str, bool, torch.Tensor, AAFrame]]
# =============================================================================
# Functions
# =============================================================================
# =============================================================================
# Classes
# =============================================================================
# =============================================================================
# Tests
# =============================================================================
if __name__ == '__main__':
    pass
