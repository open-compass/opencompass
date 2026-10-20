import csv
import json
import os
import re
from typing import Dict, Iterable, Optional, Tuple

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

ATOM_COLUMNS = [
    'carbon', 'oxygen', 'nitrogen', 'sulfur', 'fluorine', 'chlorine',
    'bromine', 'iodine', 'phosphorus', 'boron', 'silicon', 'selenium',
    'tellurium', 'arsenic', 'antimony', 'bismuth', 'polonium'
]

ATOM_SYMBOLS = {
    'carbon': 'C',
    'oxygen': 'O',
    'nitrogen': 'N',
    'sulfur': 'S',
    'fluorine': 'F',
    'chlorine': 'Cl',
    'bromine': 'Br',
    'iodine': 'I',
    'phosphorus': 'P',
    'boron': 'B',
    'silicon': 'Si',
    'selenium': 'Se',
    'tellurium': 'Te',
    'arsenic': 'As',
    'antimony': 'Sb',
    'bismuth': 'Bi',
    'polonium': 'Po',
}

BOND_COLUMNS = ['single', 'double', 'triple', 'rotatable', 'aromatic']

FUNCTIONAL_GROUP_COLUMNS = [
    'benzene_ring', 'hydroxyl', 'anhydride', 'aldehyde', 'ketone', 'carboxyl',
    'ester', 'amide', 'amine', 'nitro', 'halo', 'thioether', 'nitrile',
    'thiol', 'sulfide', 'disulfide', 'sulfoxide', 'sulfone', 'borane'
]

FUNCTIONAL_GROUP_SMARTS = {
    'benzene_ring': ['[cR1]1[cR1][cR1][cR1][cR1][cR1]1'],
    'hydroxyl': ['[OX2H]'],
    'anhydride': ['[CX3](=[OX1])[OX2][CX3](=[OX1])'],
    'aldehyde': ['[CX3H1](=O)[#6]'],
    'ketone': ['[#6][CX3](=[OX1])[#6]'],
    'carboxyl': ['[CX3](=O)[OX2H1]'],
    'ester': ['[#6][CX3](=O)[OX2H0][#6]'],
    'amide': ['[NX3][CX3](=[OX1])[#6]'],
    'amine': ['[NX3;H2,H1;!$(NC=O)]'],
    'nitro': ['[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]'],
    'halo': ['[F,Cl,Br,I]'],
    'thioether': ['[SX2][CX4]'],
    'nitrile': ['[NX1]#[CX2]'],
    'thiol': ['[#16X2H]'],
    'sulfide': ['[#16X2H0]'],
    'disulfide': ['[#16X2H0][#16X2H0]'],
    'sulfoxide': ['[$([#16X3]=[OX1]),$([#16X3+][OX1-])]'],
    'sulfone': ['[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]'],
    'borane': ['[BX3]'],
}

TASKS = {
    'MolCustom_AtomNum': ('MolCustom', 'AtomNum'),
    'MolCustom_BondNum': ('MolCustom', 'BondNum'),
    'MolCustom_FunctionalGroup': ('MolCustom', 'FunctionalGroup'),
    'MolEdit_AddComponent': ('MolEdit', 'AddComponent'),
    'MolEdit_DelComponent': ('MolEdit', 'DelComponent'),
    'MolEdit_SubComponent': ('MolEdit', 'SubComponent'),
    'MolOpt_LogP': ('MolOpt', 'LogP'),
    'MolOpt_MR': ('MolOpt', 'MR'),
    'MolOpt_QED': ('MolOpt', 'QED'),
}

MAX_SMILES_CHARS = 2048


def _safe_int(value) -> int:
    if value in (None, ''):
        return 0
    return int(float(value))


def _safe_float(value) -> float:
    if value in (None, ''):
        return 0.0
    return float(value)


def _read_csv_rows(csv_path: str) -> Iterable[Dict[str, str]]:
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        yield from csv.DictReader(f)


def _correct_prediction_text(text: str) -> str:
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            json_str = match.group().replace('""', '"')
            try:
                data = json.loads(json_str)
                candidate = data.get('molecule', json_str)
            except Exception:
                candidate = json_str.split(':')[1].strip().strip('}').strip()
                candidate = candidate.strip('"').strip()
            for sep in ('=>', '->'):
                if sep in candidate:
                    candidate = candidate.split(sep)[-1].strip()
            return candidate.strip()

        candidate = text.replace('\n', ' ').strip()
        for sep in ('=>', '->'):
            if sep in candidate:
                candidate = candidate.split(sep)[-1].strip()
        if len(candidate
               ) >= 2 and candidate[0] == '[' and candidate[-1] == ']':
            candidate = candidate[1:-1]
        return candidate
    except Exception:
        return 'None'


def extract_smiles(prediction: str) -> Optional[str]:
    if isinstance(prediction, list) and prediction:
        prediction = prediction[0]
    if not isinstance(prediction, str):
        return None

    candidate = _correct_prediction_text(prediction)
    if not candidate or len(candidate) > MAX_SMILES_CHARS:
        return None
    try:
        from rdkit import Chem  # noqa: F401
    except Exception:
        return candidate.strip() or None

    mol = _rdkit_mol_from_smiles(candidate)
    if mol is not None:
        return candidate.strip()
    return None


def _rdkit_mol_from_smiles(smiles: str):
    from rdkit import Chem, RDLogger

    if not smiles or len(smiles) > MAX_SMILES_CHARS:
        return None
    RDLogger.DisableLog('rdApp.error')
    try:
        return Chem.MolFromSmiles(smiles)
    finally:
        RDLogger.EnableLog('rdApp.error')


def _mol_from_smiles(smiles: Optional[str]):
    if not smiles:
        return None
    return _rdkit_mol_from_smiles(smiles)


def _fingerprint_similarity(pred_mol, source_mol) -> float:
    if pred_mol is None or source_mol is None:
        return 0.0
    from rdkit.Chem import DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    generator = GetMorganGenerator(radius=2, fpSize=2048)
    pred_fp = generator.GetFingerprint(pred_mol)
    source_fp = generator.GetFingerprint(source_mol)
    return float(DataStructs.TanimotoSimilarity(pred_fp, source_fp))


def _atom_counts(mol) -> Dict[str, int]:
    counts = {name: 0 for name in ATOM_COLUMNS}
    if mol is None:
        return counts
    symbol_to_name = {symbol: name for name, symbol in ATOM_SYMBOLS.items()}
    for atom in mol.GetAtoms():
        name = symbol_to_name.get(atom.GetSymbol())
        if name:
            counts[name] += 1
    return counts


def _bond_counts(mol) -> Dict[str, int]:
    counts = {name: 0 for name in BOND_COLUMNS}
    if mol is None:
        return counts
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            counts['single'] += 1
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            counts['double'] += 1
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            counts['triple'] += 1
        if bond.GetIsAromatic():
            counts['aromatic'] += 1
    counts['rotatable'] = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    return counts


def _functional_group_patterns():
    from rdkit import Chem

    compiled = {}
    for name, smarts_list in FUNCTIONAL_GROUP_SMARTS.items():
        patterns = []
        for smarts in smarts_list:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                patterns.append(pattern)
        compiled[name] = patterns
    return compiled


def _functional_group_counts(mol) -> Dict[str, int]:
    counts = {name: 0 for name in FUNCTIONAL_GROUP_COLUMNS}
    if mol is None:
        return counts
    for name, patterns in _functional_group_patterns().items():
        match_sets = set()
        for pattern in patterns:
            for match in mol.GetSubstructMatches(pattern):
                match_sets.add(tuple(match))
        counts[name] = len(match_sets)
    counts['sulfide'] = max(0, counts['sulfide'] - counts['disulfide'])
    return counts


def _molecule_property(mol, property_name: str) -> float:
    if mol is None:
        return 0.0
    from rdkit.Chem import QED, Crippen

    if property_name == 'LogP':
        return float(Crippen.MolLogP(mol))
    if property_name == 'MR':
        return float(Crippen.MolMR(mol))
    if property_name == 'QED':
        return float(QED.qed(mol))
    raise ValueError(f'Unsupported S2-TOMG property: {property_name}')


def _is_increase_instruction(instruction: str) -> bool:
    text = instruction.lower()
    if 'lower' in text or 'decrease' in text:
        return False
    return True


def _target_constraints(row: Dict[str, str], columns) -> Dict[str, int]:
    return {column: _safe_int(row.get(column)) for column in columns}


@LOAD_DATASET.register_module()
class S2TOMGBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str,
             name: str,
             task_family: Optional[str] = None,
             subtask: Optional[str] = None,
             max_samples: Optional[int] = None,
             **kwargs):
        data_dir = get_data_path(path)
        filename = name if name.endswith('.csv') else f'{name}.csv'
        csv_path = os.path.join(data_dir, filename)
        inferred_family, inferred_subtask = TASKS.get(
            os.path.splitext(filename)[0], (task_family, subtask))
        task_family = task_family or inferred_family
        subtask = subtask or inferred_subtask

        rows = []
        for row_idx, item in enumerate(_read_csv_rows(csv_path)):
            reference = {
                'task_name': os.path.splitext(filename)[0],
                'task_family': task_family,
                'subtask': subtask,
                'instruction': item['Instruction'],
            }
            if task_family == 'MolCustom' and subtask == 'AtomNum':
                reference['constraints'] = _target_constraints(
                    item, ATOM_COLUMNS)
            elif task_family == 'MolCustom' and subtask == 'BondNum':
                reference['constraints'] = _target_constraints(
                    item, BOND_COLUMNS)
            elif task_family == 'MolCustom' and subtask == 'FunctionalGroup':
                reference['constraints'] = _target_constraints(
                    item, FUNCTIONAL_GROUP_COLUMNS)
            elif task_family == 'MolEdit':
                reference['source_molecule'] = item['molecule']
                if 'added_group' in item:
                    reference['added_group'] = item['added_group']
                if 'removed_group' in item:
                    reference['removed_group'] = item['removed_group']
            elif task_family == 'MolOpt':
                reference['source_molecule'] = item['molecule']
                reference['property_name'] = subtask
                source_property = item.get(subtask)
                if source_property is None and subtask == 'LogP':
                    source_property = item.get('logP')
                reference['source_property'] = (_safe_float(source_property)
                                                if source_property
                                                not in (None, '') else None)
                reference['increase'] = _is_increase_instruction(
                    item['Instruction'])
            else:
                raise ValueError(
                    f'Unsupported S2-TOMG task: {task_family}/{subtask}')

            rows.append({
                'id': item.get('index') or str(row_idx),
                'prompt': item['Instruction'],
                'reference': reference,
                'task_name': reference['task_name'],
            })
            if max_samples and len(rows) >= max_samples:
                break
        return Dataset.from_list(rows)


@ICL_EVALUATORS.register_module()
class S2TOMGBenchEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        details = []
        valid_count = 0
        success_values = []
        weighted_values = []
        similarities = []

        for pred_raw, ref in zip(predictions, references):
            smiles = extract_smiles(pred_raw)
            pred_mol = _mol_from_smiles(smiles)
            valid = pred_mol is not None
            if valid:
                valid_count += 1

            success, weight, similarity, extra = self._score_one(pred_mol, ref)
            success_values.append(1.0 if success else 0.0)
            weighted_values.append(weight)
            if similarity is not None:
                similarities.append(similarity)

            details.append({
                'prediction': pred_raw,
                'smiles': smiles,
                'valid': int(valid),
                'success': int(success),
                'weight': weight,
                'similarity': similarity,
                'reference': ref,
                **extra,
            })

        n = len(predictions)
        result = {
            'score': sum(weighted_values) / n * 100 if n else 0.0,
            'success_rate': sum(success_values) / n * 100 if n else 0.0,
            'validity': valid_count / n * 100 if n else 0.0,
            'weighted_success_rate':
            sum(weighted_values) / n * 100 if n else 0.0,
            'details': details,
        }
        if similarities:
            result['similarity'] = sum(similarities) / len(similarities) * 100
        return result

    def _score_one(self, pred_mol,
                   ref: Dict) -> Tuple[bool, float, Optional[float], Dict]:
        if pred_mol is None:
            return False, 0.0, None, {'reason': 'invalid_smiles'}

        family = ref['task_family']
        subtask = ref['subtask']
        if family == 'MolCustom':
            success, observed = self._score_custom(pred_mol, subtask,
                                                   ref['constraints'])
            # The official repo can optionally calculate novelty against an
            # external reference corpus. The local benchmark CSV does not carry
            # that corpus, so custom-task WSR falls back to SR and records this
            # caveat in details.
            return success, 1.0 if success else 0.0, None, {
                'observed': observed,
                'novelty_reference': 'missing',
            }

        source_mol = _mol_from_smiles(ref.get('source_molecule'))
        similarity = _fingerprint_similarity(pred_mol, source_mol)
        if family == 'MolEdit':
            success, observed = self._score_edit(pred_mol, source_mol, subtask,
                                                 ref)
            return success, similarity if success else 0.0, similarity, {
                'observed': observed,
            }

        if family == 'MolOpt':
            prop = ref['property_name']
            pred_value = _molecule_property(pred_mol, prop)
            source_value = ref.get('source_property')
            if source_value is None and source_mol is not None:
                source_value = _molecule_property(source_mol, prop)
            increase = bool(ref.get('increase', True))
            success = (pred_value > source_value
                       if increase else pred_value < source_value)
            return success, similarity if success else 0.0, similarity, {
                'pred_property': pred_value,
                'source_property': source_value,
                'increase': increase,
            }

        raise ValueError(f'Unsupported S2-TOMG family: {family}')

    @staticmethod
    def _score_custom(
            pred_mol, subtask: str,
            constraints: Dict[str, int]) -> Tuple[bool, Dict[str, int]]:
        if subtask == 'AtomNum':
            observed = _atom_counts(pred_mol)
        elif subtask == 'BondNum':
            observed = _bond_counts(pred_mol)
        elif subtask == 'FunctionalGroup':
            observed = _functional_group_counts(pred_mol)
        else:
            raise ValueError(f'Unsupported S2-TOMG custom subtask: {subtask}')
        if subtask == 'BondNum':
            success = all(value == 0 or observed.get(key, 0) == value
                          for key, value in constraints.items())
        elif subtask == 'FunctionalGroup':
            success = all(key == 'thioether' or observed.get(key, 0) == value
                          for key, value in constraints.items())
        else:
            success = all(
                observed.get(key, 0) == value
                for key, value in constraints.items())
        return success, observed

    @staticmethod
    def _score_edit(pred_mol, source_mol, subtask: str,
                    ref: Dict) -> Tuple[bool, Dict[str, Dict[str, int]]]:
        pred_counts = _functional_group_counts(pred_mol)
        source_counts = _functional_group_counts(source_mol)
        added_group = ref.get('added_group')
        removed_group = ref.get('removed_group')

        added_ok = True
        removed_ok = True
        if added_group:
            added_key = added_group.replace(' ', '_')
            added_ok = pred_counts.get(
                added_key, 0) == source_counts.get(added_key, 0) + 1
        if removed_group:
            removed_key = removed_group.replace(' ', '_')
            removed_ok = pred_counts.get(
                removed_key, 0) == source_counts.get(removed_key, 0) - 1

        if subtask == 'AddComponent':
            success = added_ok
        elif subtask == 'DelComponent':
            success = removed_ok
        elif subtask == 'SubComponent':
            success = added_ok and removed_ok
        else:
            raise ValueError(f'Unsupported S2-TOMG edit subtask: {subtask}')

        return success, {
            'pred_counts': pred_counts,
            'source_counts': source_counts,
        }
