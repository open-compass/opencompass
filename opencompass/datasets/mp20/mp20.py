import ast
import itertools
import json
import re
from multiprocessing import Pool
from multiprocessing import TimeoutError as MPTimeoutError

import numpy as np
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset

LTOL = 0.3
STOL = 0.5
ANGLE_TOL = 10


@LOAD_DATASET.register_module()
class MP20Dataset(BaseDataset):

    @staticmethod
    def load(path: str, **kwargs):
        path = get_data_path(path)
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                samples.append({
                    'id_ddm': d['id_ddm'],
                    'system_prompt': d['system_prompt'],
                    'user_input': d['user_input'],
                    'ground_truth': d['ground_truth'],
                })
        return Dataset.from_list(samples)


# JSON extraction helpers


def _extract_last_large_json(text, min_length=100):
    decoder = json.JSONDecoder()
    valid_jsons = []
    parsed_end = 0
    for m in re.finditer(r'\{|\[', text):
        start_idx = m.start()
        if start_idx < parsed_end:
            continue
        try:
            parsed_obj, offset = decoder.raw_decode(text[start_idx:])
            end_idx = start_idx + offset
            json_str = text[start_idx:end_idx]
            if len(json_str) >= min_length:
                valid_jsons.append(json_str)
            parsed_end = end_idx
        except json.JSONDecodeError:
            pass
    return valid_jsons[-1] if valid_jsons else None


def _robust_parse_json(text):
    for tag in [
            '<lattice>', '</lattice>', '<sites>', '</sites>', '<material>',
            '</material>', '</think>'
    ]:
        text = text.replace(tag, '')

    start_idx = text.find('{')
    if start_idx == -1:
        raise ValueError("No '{' found in text")
    clean_text = text[start_idx:].strip()

    try:
        parsed_obj, _ = json.JSONDecoder().raw_decode(clean_text)
        return parsed_obj
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', clean_text, re.DOTALL)
    if not match:
        raise ValueError('Cannot find braces structure')
    fallback_text = match.group(0)

    try:
        return ast.literal_eval(fallback_text)
    except Exception:
        pass

    fixed = fallback_text.replace("'", '"')
    fixed = re.sub(r'\bTrue\b', 'true', fixed)
    fixed = re.sub(r'\bFalse\b', 'false', fixed)
    fixed = re.sub(r'\bNone\b', 'null', fixed)
    try:
        parsed_obj, _ = json.JSONDecoder().raw_decode(fixed)
        return parsed_obj
    except Exception as e:
        raise ValueError(f'Parse failed: {e}\n{clean_text[:100]}...')


def _lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    alpha_r, beta_r, gamma_r = np.radians(alpha), np.radians(beta), np.radians(
        gamma)
    val = (np.cos(alpha_r) -
           np.cos(beta_r) * np.cos(gamma_r)) / np.sin(gamma_r)
    z_val_sq = max(0.0, 1 - np.cos(beta_r)**2 - val**2)
    return [
        [a, 0, 0],
        [b * np.cos(gamma_r), b * np.sin(gamma_r), 0],
        [c * np.cos(beta_r), c * val, c * np.sqrt(z_val_sq)],
    ]


_MATERIAL_PAT = re.compile(r'<material>(.*?)</material>', re.DOTALL)
_MATERIAL_PAT2 = re.compile(r'output:\n(.*?)\n </think>', re.DOTALL)
_MATERIAL_PAT3 = re.compile(r'```json(.*?)```', re.DOTALL)


def _extract_prediction(raw_output):
    m1 = _MATERIAL_PAT.search(raw_output)
    m2 = _MATERIAL_PAT2.search(raw_output)
    m3 = _MATERIAL_PAT3.search(raw_output)

    if not m1 and not m2 and not m3:
        raw_material_text = _extract_last_large_json(raw_output, min_length=50)
        if raw_material_text is None:
            raise ValueError('No extractable JSON')
    else:
        raw_material_text = m1.group(1) if m1 else None
        if m2:
            raw_material_text = m2.group(1)
        if m3:
            raw_material_text = m3.group(1)

    pred_dict = _robust_parse_json(raw_material_text)
    if not isinstance(pred_dict, dict):
        raise ValueError('Parsed object is not a dict')

    lat = pred_dict.get('lattice_parameters', {})
    a = float(lat.get('a', 0))
    b = float(lat.get('b', 0))
    c = float(lat.get('c', 0))
    alpha = float(lat.get('alpha', 90))
    beta = float(lat.get('beta', 90))
    gamma = float(lat.get('gamma', 90))
    pred_lattice = _lattice_params_to_matrix(a, b, c, alpha, beta, gamma)

    pred_sites = pred_dict.get('atomic_sites', [])
    pred_coords = [[float(x) for x in site['coordinate']]
                   for site in pred_sites]
    return pred_lattice, pred_coords


# Validity helpers


def _smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    import smact
    from smact.screening import pauling_test

    space = smact.element_dictionary(comp)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(comp)) == 1:
        return True
    if include_alloys and all(e in smact.metals for e in comp):
        return True
    threshold = np.max(count)
    compositions = []
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c, ) for c in count]
        cn_e, cn_r = smact.neutral_ratios(ox_states,
                                          stoichs=stoichs,
                                          threshold=threshold)
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(tuple([comp, ox_states, ratio]))
    compositions = list(set((i[0], i[2]) for i in compositions))
    return len(compositions) > 0


def _structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix + np.diag(
        np.ones(crystal.distance_matrix.shape[0]) * (cutoff + 10.0))
    return dist_mat.min() >= cutoff and crystal.volume >= 0.1


# RMS distance with subprocess timeout

_worker_pool = None


def _get_worker_pool():
    global _worker_pool
    if _worker_pool is None:
        _worker_pool = Pool(1)
    return _worker_pool


def _rms_dist_worker(pred_lattice, pred_coords, gt_lattice, species,
                     gt_frac_coords):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core.lattice import Lattice
    from pymatgen.core.structure import Structure
    matcher = StructureMatcher(stol=STOL, angle_tol=ANGLE_TOL, ltol=LTOL)
    pred = Structure(Lattice(pred_lattice), species, pred_coords)
    gt = Structure(Lattice(gt_lattice), species, gt_frac_coords)
    result = matcher.get_rms_dist(pred, gt)
    return None if result is None else result[0]


def _get_rms_dist_safe(pred_lattice,
                       pred_coords,
                       gt_lattice,
                       species,
                       gt_frac_coords,
                       timeout_sec=10):
    global _worker_pool
    pool = _get_worker_pool()
    async_result = pool.apply_async(
        _rms_dist_worker,
        (pred_lattice, pred_coords, gt_lattice, species, gt_frac_coords),
    )
    try:
        return async_result.get(timeout=timeout_sec)
    except MPTimeoutError:
        pool.terminate()
        _worker_pool = None
        return None
    except Exception:
        return None


# Evaluator


class MP20Evaluator(BaseEvaluator):

    def score(self, predictions, references):
        from pymatgen.core.lattice import Lattice
        from pymatgen.core.structure import Structure

        rms_dists = []

        for pred_text, gt in zip(predictions, references):
            gt_lattice = gt['lattice']
            gt_sites = gt['sites']
            species = [s['element'] for s in gt_sites]
            gt_frac_coords = [s['fractional_coordinates'] for s in gt_sites]

            try:
                pred_lattice, pred_coords = _extract_prediction(pred_text)
            except Exception:
                rms_dists.append(None)
                continue

            try:
                Structure(Lattice(pred_lattice), species, pred_coords)
                Structure(Lattice(gt_lattice), species, gt_frac_coords)
            except Exception:
                rms_dists.append(None)
                continue

            rms = _get_rms_dist_safe(pred_lattice, pred_coords, gt_lattice,
                                     species, gt_frac_coords)
            rms_dists.append(rms)

        global _worker_pool
        if _worker_pool is not None:
            _worker_pool.terminate()
            _worker_pool.join()
            _worker_pool = None

        total = len(rms_dists)
        matched = [r for r in rms_dists if r is not None]
        match_rate = len(matched) / total * 100 if total else 0.0
        mean_rms = float(np.mean(matched)) * 100 if matched else float('nan')

        return {
            'match_rate': round(match_rate, 4),
            'mean_rms_dist':
            round(mean_rms, 4) if not np.isnan(mean_rms) else None,
            'matched': len(matched),
            'total': total,
        }
