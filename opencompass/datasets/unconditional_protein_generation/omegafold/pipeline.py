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
This file contains the utilities that we use for the entire inference pipeline
"""
# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import collections
import logging
import ntpath
import os
import os.path
import pathlib
import types
import typing

import torch
from Bio import PDB as PDB
from Bio.PDB import StructureBuilder
from huggingface_hub import hf_hub_download
from torch import hub
from torch.backends import cuda, cudnn

from . import utils
from .utils.protein_utils import residue_constants as rc

try:
    from torch.backends import mps  # Compatibility with earlier versions

    _mps_is_available = mps.is_available
except ImportError:

    def _mps_is_available():
        return False


# =============================================================================
# Constants
# =============================================================================
# =============================================================================
# Functions
# =============================================================================
def _set_precision(allow_tf32: bool) -> None:
    """Set precision (mostly to do with tensorfloat32)

    This allows user to go to fp32

    Args:
        allow_tf32: if allowing

    Returns:

    """
    if int(torch.__version__.split('.')[1]) < 12:
        cuda.matmul.allow_tf32 = allow_tf32
        cudnn.allow_tf32 = allow_tf32
    else:
        precision = 'high' if allow_tf32 else 'highest'
        torch.set_float32_matmul_precision(precision)


def path_leaf(path: str) -> str:
    """
    Get the filename from the path

    Args:
        path: the absolute or relative path to the file

    Returns:
        the filename

    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def fasta2inputs(
    fasta_path: str,
    output_dir: typing.Optional[str] = None,
    num_pseudo_msa: int = 15,
    device: typing.Optional[torch.device] = torch.device('cpu'),
    mask_rate: float = 0.12,
    num_cycle: int = 10,
    deterministic: bool = True
) -> typing.Generator[typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                   str], None, None]:
    """
    Load a fasta file and

    Args:
        fasta_path: the path to the fasta files
        output_dir: the path to the output directory
        num_pseudo_msa:
        device: the device to move
        mask_rate:
        num_cycle:
        deterministic:

    Returns:

    """
    chain_ids: list[str] = []
    aastr: list[str] = []
    with open(fasta_path, 'r') as file:
        lines = file.readlines()
    name = False
    for line in lines:
        if len(line) == 0:
            continue
        if line.startswith('>') or line.startswith(':'):
            name = True
            chain_ids.append(line[1:].strip('\n'))
        else:
            if name:
                aastr.append(line.strip('\n').upper())
                name = False
            else:
                aastr[-1] = aastr[-1] + line.strip('\n').upper()

    combined = sorted(list(zip(chain_ids, aastr)), key=lambda x: len(x[1]))
    if output_dir is None:
        parent = pathlib.Path(fasta_path).parent
        folder_name = path_leaf(fasta_path).split('.')[0]
        output_dir = os.path.join(parent, folder_name)
        os.makedirs(output_dir, exist_ok=True)
    try:
        name_max = os.pathconf(output_dir, 'PC_NAME_MAX') - 4
    except AttributeError:
        # os.pathconf is UNIX specific. Set to 32 for now.
        name_max = 32

    for i, (ch, fas) in enumerate(combined):
        fas = fas.replace('Z', 'E').replace('B', 'D').replace('U', 'C')
        aatype = torch.LongTensor(
            [rc.restypes_with_x.index(aa) if aa != '-' else 21 for aa in fas])
        mask = torch.ones_like(aatype).float()
        assert torch.all(aatype.ge(0)) and torch.all(aatype.le(21)), \
            'Only take 0-20 amino acids as inputs with unknown amino acid ' \
            'indexed as 20'
        if len(ch) < name_max:
            out_fname = ch.replace(os.path.sep, '-')
        else:
            out_fname = f'{i}th chain'
        out_fname = os.path.join(output_dir, out_fname + '.pdb')

        num_res = len(aatype)
        data = list()
        g = None
        if deterministic:
            g = torch.Generator()
            g.manual_seed(num_res)
        for _ in range(num_cycle):
            p_msa = aatype[None, :].repeat(num_pseudo_msa, 1)
            p_msa_mask = torch.rand([num_pseudo_msa, num_res],
                                    generator=g).gt(mask_rate)
            p_msa_mask = torch.cat((mask[None, :], p_msa_mask), dim=0)
            p_msa = torch.cat((aatype[None, :], p_msa), dim=0)
            p_msa[~p_msa_mask.bool()] = 21
            data.append({'p_msa': p_msa, 'p_msa_mask': p_msa_mask})

        yield utils.recursive_to(data, device=device), out_fname


# modify fasta2inputs to list2inputs
def list2inputs(
    protein_list: typing.List[str],
    output_dir: typing.Optional[str] = None,
    num_pseudo_msa: int = 15,
    device: typing.Optional[torch.device] = torch.device('cpu'),
    mask_rate: float = 0.12,
    num_cycle: int = 10,
    deterministic: bool = True
) -> typing.Generator[typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                   str], None, None]:
    """
    Load a fasta file and

    Args:
        fasta_path: the path to the fasta files
        output_dir: the path to the output directory
        num_pseudo_msa:
        device: the device to move
        mask_rate:
        num_cycle:
        deterministic:

    Returns:

    """
    chain_ids: list[str] = []
    aastr: list[str] = []

    chain_ids = [f'chain_{i}' for i in range(len(protein_list))]
    aastr = protein_list

    combined = sorted(list(zip(chain_ids, aastr)), key=lambda x: len(x[1]))
    name_max = 32

    for i, (ch, fas) in enumerate(combined):
        fas = fas.replace('Z', 'E').replace('B', 'D').replace('U', 'C')
        aatype = torch.LongTensor(
            [rc.restypes_with_x.index(aa) if aa != '-' else 21 for aa in fas])
        mask = torch.ones_like(aatype).float()
        assert torch.all(aatype.ge(0)) and torch.all(aatype.le(21)), \
            'Only take 0-20 amino acids as inputs with unknown amino acid ' \
            'indexed as 20'
        if len(ch) < name_max:
            out_fname = ch.replace(os.path.sep, '-')
        else:
            out_fname = f'{i}th chain'
        out_fname = os.path.join(output_dir, out_fname + '.pdb')

        num_res = len(aatype)
        data = list()
        g = None
        if deterministic:
            g = torch.Generator()
            g.manual_seed(num_res)
        for _ in range(num_cycle):
            p_msa = aatype[None, :].repeat(num_pseudo_msa, 1)
            p_msa_mask = torch.rand([num_pseudo_msa, num_res],
                                    generator=g).gt(mask_rate)
            p_msa_mask = torch.cat((mask[None, :], p_msa_mask), dim=0)
            p_msa = torch.cat((aatype[None, :], p_msa), dim=0)
            p_msa[~p_msa_mask.bool()] = 21
            data.append({'p_msa': p_msa, 'p_msa_mask': p_msa_mask})

        yield utils.recursive_to(data, device=device), out_fname


def save_pdb(pos14: torch.Tensor,
             b_factors: torch.Tensor,
             sequence: torch.Tensor,
             mask: torch.Tensor,
             save_path: str,
             model: int = 0,
             init_chain: str = 'A') -> None:
    """
    saves the pos14 as a pdb file

    Args:
        pos14: the atom14 representation of the coordinates
        b_factors: the b_factors of the amino acids
        sequence: the amino acid of the pos14
        mask: the validity of the atoms
        save_path: the path to save the pdb file
        model: the model id of the pdb file
        init_chain

    return:
        the structure saved to ~save_path

    """
    builder = StructureBuilder.StructureBuilder()
    builder.init_structure(0)
    builder.init_model(model)
    builder.init_chain(init_chain)
    builder.init_seg('    ')
    for i, (aa_idx, p_res, b,
            m_res) in enumerate(zip(sequence, pos14, b_factors, mask.bool())):
        if not m_res:
            continue
        aa_idx = aa_idx.item()
        p_res = p_res.clone().detach().cpu()
        if aa_idx == 21:
            continue
        try:
            three = rc.residx_to_3(aa_idx)
        except IndexError:
            continue
        builder.init_residue(three, ' ', int(i), icode=' ')
        for j, (atom_name, ) in enumerate(
                zip(rc.restype_name_to_atom14_names[three])):
            if len(atom_name) > 0:
                builder.init_atom(atom_name,
                                  p_res[j].tolist(),
                                  b.item(),
                                  1.0,
                                  ' ',
                                  atom_name.join([' ', ' ']),
                                  element=atom_name[0])
    structure = builder.get_structure()
    io = PDB.PDBIO()
    io.set_structure(structure)
    os.makedirs(pathlib.Path(save_path).parent, exist_ok=True)
    io.save(save_path)


def _load_weights(
    weights_url: str,
    weights_file: str,
) -> collections.OrderedDict:
    """
    Loads the weights from either a url or a local file. If from url,

    Args:
        weights_url: a url for the weights
        weights_file: a local file

    Returns:
        state_dict: the state dict for the model

    """

    weights_file = os.path.expanduser(weights_file)
    use_cache = os.path.exists(weights_file)
    if weights_file and weights_url and not use_cache:
        logging.info(
            f'Downloading weights from {weights_url} to {weights_file}')
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)
        hub.download_url_to_file(weights_url, weights_file)
    else:
        logging.info(f'Loading weights from {weights_file}')

    return torch.load(weights_file, map_location='cpu')


def _get_device(device) -> str:
    """
    Infer the accelerator

    Args:
        device: the device type

    Returns:

    """
    if device is None:
        if torch.cuda.is_available():
            return 'cuda'
        elif _mps_is_available():
            return 'mps'
        else:
            return 'cpu'
    elif device == 'cpu':
        return device
    elif device.startswith('cuda'):
        if torch.cuda.is_available():
            return device
        else:
            raise ValueError('Device cuda is not available')
    elif device == 'mps':
        if _mps_is_available():
            return device
        else:
            raise ValueError('Device mps is not available')
    else:
        raise ValueError(f'Device type {device} is not available')


def get_args() -> typing.Tuple[types.SimpleNamespace, collections.OrderedDict,
                               types.SimpleNamespace]:

    # 直接构造 args 对象，代替 argparse.Namespace
    args = types.SimpleNamespace()
    args.num_cycle = 10
    args.subbatch_size = 448
    args.device = None
    # TODO: Modify the path of weights_file
    args.weights_file = hf_hub_download('SciReason/OmegaFold-release',
                                        'release2.pt',
                                        repo_type='dataset')
    args.weights = 'https://helixon.s3.amazonaws.com/release1.pt'
    args.model = 2
    args.pseudo_msa_mask_rate = 0.12
    args.num_pseudo_msa = 15
    args.allow_tf32 = True

    _set_precision(args.allow_tf32)

    if args.model == 1:
        weights_url = 'https://helixon.s3.amazonaws.com/release1.pt'
        if args.weights_file is None:
            args.weights_file = os.path.expanduser(
                '~/.cache/omegafold_ckpt/model.pt')
    elif args.model == 2:
        weights_url = 'https://helixon.s3.amazonaws.com/release2.pt'
        if args.weights_file is None:
            args.weights_file = os.path.expanduser(
                '~/.cache/omegafold_ckpt/model2.pt')
    else:
        raise ValueError(
            f'Model {args.model} is not available, only 1 or 2 supported.')

    # 加载权重
    weights = _load_weights(weights_url, args.weights_file)
    weights = weights.pop('model', weights)

    # 构造 forward_config
    forward_config = types.SimpleNamespace(
        subbatch_size=args.subbatch_size,
        num_recycle=args.num_cycle,
    )

    # 自动设置设备
    args.device = _get_device(args.device)

    return args, weights, forward_config


# =============================================================================
# Classes
# =============================================================================
# =============================================================================
# Tests
# =============================================================================
if __name__ == '__main__':
    pass
