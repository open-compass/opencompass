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
Static configuration reside in this file
"""
# =============================================================================
# Imports
# =============================================================================
import argparse


# =============================================================================
# Constants
# =============================================================================
# =============================================================================
# Functions
# =============================================================================
def _make_config(input_dict: dict) -> argparse.Namespace:
    """Recursively go through dictionary"""
    new_dict = {}
    for k, v in input_dict.items():
        if type(v) == dict:
            new_dict[k] = _make_config(v)
        else:
            new_dict[k] = v
    return argparse.Namespace(**new_dict)


def make_config(model_idx: int = 1) -> argparse.Namespace:
    if model_idx not in [1, 2]:
        raise ValueError('model_idx must be 1 or 2')
    cfg = dict(alphabet_size=21,
               plm=dict(
                   alphabet_size=23,
                   node=1280,
                   padding_idx=21,
                   edge=66,
                   proj_dim=1280 * 2,
                   attn_dim=256,
                   num_head=1,
                   num_relpos=129,
                   masked_ratio=0.12,
               ),
               node_dim=256,
               edge_dim=128,
               relpos_len=32,
               prev_pos=dict(
                   first_break=3.25,
                   last_break=20.75,
                   num_bins=16,
                   ignore_index=0,
               ),
               rough_dist_bin=dict(
                   x_min=3.25,
                   x_max=20.75,
                   x_bins=16,
               ),
               dist_bin=dict(
                   x_bins=64,
                   x_min=2,
                   x_max=65,
               ),
               pos_bin=dict(
                   x_bins=64,
                   x_min=-32,
                   x_max=32,
               ),
               c=16,
               geo_num_blocks=50,
               gating=True,
               attn_c=32,
               attn_n_head=8,
               transition_multiplier=4,
               activation='ReLU',
               opm_dim=32,
               geom_count=2,
               geom_c=32,
               geom_head=4,
               struct=dict(
                   node_dim=384,
                   edge_dim=128,
                   num_cycle=8,
                   num_transition=3,
                   num_head=12,
                   num_point_qk=4,
                   num_point_v=8,
                   num_scalar_qk=16,
                   num_scalar_v=16,
                   num_channel=128,
                   num_residual_block=2,
                   hidden_dim=128,
                   num_bins=50,
               ))
    cfg['struct_embedder'] = model_idx == 2
    return _make_config(cfg)


# =============================================================================
# Classes
# =============================================================================
# =============================================================================
# Tests
# =============================================================================
if __name__ == '__main__':
    pass
