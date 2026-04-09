"""
Reward functions for constraint generation tasks.
Uses solver.py to verify that generated molecules satisfy constraints.

Supported constraint types (from the comprehensive list):

RING PROPERTIES:
- ring, fused_ring, spiro, bridgehead
- smallest_ring_size, largest_ring_size
- aromatic_ring, aliphatic_ring, heterocycle, saturated_ring

CHAIN PROPERTIES:
- chain_termini, branch_point
- longest_carbon_chain, csp3_carbon

STEREOCHEMISTRY:
- stereocenter, r_s_stereocenter, unspecified_stereocenter
- e_z_stereochemistry_double_bond, stereochemistry_unspecified_double_bond

ATOM COUNTS:
- carbon_atom, hetero_atom, halogen_atom
- heavy_atom, hydrogen_atom

MOLECULAR PROPERTIES:
- molecular_formula, hba, hbd, rotatable_bond

ADVANCED PROPERTIES:
- oxidation_state (various element/state combinations)
- functional_group (135+ functional groups from smarts_renamed.txt)
- brics_decomposition, murcko_scaffold
- template_based_reaction_prediction
"""

import json
from functools import lru_cache
from typing import Any, Dict, List, Union

from .utils import (
    are_same_molecular_formula,
    evaluate_numeric_constraint,
    is_reasonable_molecule,
    parse_natural_language_property,
    valid_smiles,
)

from ..solver.solver import SymbolicSolver


_SOLVER = SymbolicSolver()
_UNSUPPORTED = object()


def _extract_smiles_prediction(predicted: Union[str, Dict[str, Any], List[Any]]) -> str:
    """Normalize the predicted result into a SMILES string."""

    def _from_dict(data: Dict[str, Any]) -> Any:
        # Prefer explicit 'smiles' key (case insensitive)
        for key in data.keys():
            if key.lower() == 'smiles':
                return data[key]
        # Fallback: if single entry, return its value
        if len(data) == 1:
            return next(iter(data.values()))
        return data

    def _from_list(items: List[Any]) -> Any:
        if not items:
            return ''
        if len(items) == 1:
            return items[0]
        return items

    current = predicted

    if isinstance(current, dict):
        current = _from_dict(current)

    if isinstance(current, list):
        current = _from_list(current)

    if isinstance(current, str):
        text = current.strip()
        if not text:
            return text

        # Try JSON object
        if text.startswith('{') and text.endswith('}'):
            try:
                parsed = json.loads(text)
                return _extract_smiles_prediction(parsed)
            except json.JSONDecodeError:
                pass

        # Try "smiles: ..." pattern
        lower = text.lower()
        if lower.startswith('smiles:'):
            return text.split(':', 1)[1].strip()

        return text

    # Fallback to string conversion
    return '' if current is None else str(current)


@lru_cache(maxsize=128)
def _functional_group_data(smiles: str) -> Dict[str, Dict[str, Any]]:
    return _SOLVER.get_functional_group_count_and_indices(smiles)


@lru_cache(maxsize=128)
def _reaction_data(smiles: str) -> Dict[str, Dict[str, Any]]:
    return _SOLVER.get_reaction_counts_and_indices(smiles)


def _functional_group_instances(smiles: str, group_name: str) -> int:
    data = _functional_group_data(smiles)
    # The data has keys like 'functional_group_ester_count', 'functional_group_ester_nbrInstances'
    # We want the nbrInstances value (number of instances of the functional group)
    key = f"functional_group_{group_name}_nbrInstances"
    if key in data:
        return data[key]
    # Fallback to count if nbrInstances not available
    key = f"functional_group_{group_name}_count"
    if key in data:
        return data[key]
    return 0


def _reaction_template_count(smiles: str, template_name: str) -> int:
    """Get count for a reaction template.

    The template_name should be the base name like 'alcohol_to_tosylate'.
    We'll look for 'template_based_reaction_prediction_{template_name}_count' in the data.
    """
    data = _reaction_data(smiles)

    # Try different key formats
    # First try the exact template name
    if template_name in data:
        entry = data[template_name]
        if isinstance(entry, dict):
            return entry.get('count', 0)
        elif isinstance(entry, (int, float)):
            return int(entry)

    # Try with full prefix if not already present
    if not template_name.startswith('template_based_reaction_prediction_'):
        full_key = f'template_based_reaction_prediction_{template_name}_count'
        if full_key in data:
            return data[full_key]

        # Also try success key (for constraint checking)
        success_key = f'template_based_reaction_prediction_{template_name}_success'
        if success_key in data:
            return data[success_key]

    return 0


def _call_solver(smiles: str, candidates: List[str]) -> Any:
    for candidate in candidates:
        method = getattr(_SOLVER, f"get_{candidate}", None)
        if callable(method):
            try:
                return method(smiles)
            except Exception:
                return None
    return _UNSUPPORTED


def _resolve_constraint_value(smiles: str, constraint: Dict[str, Any]) -> Any:
    # Import the mapping
    from .property_solver_mapping import (
        get_solver_mapping,
        is_string_valued_property
    )

    # Get property name from either 'type' or 'property' field
    raw_type = constraint.get('type', constraint.get('property', ''))
    if not raw_type:
        return _UNSUPPORTED

    normalized = parse_natural_language_property(str(raw_type)).strip()
    if not normalized:
        return _UNSUPPORTED

    # ============================================================================
    # USE EXPLICIT PROPERTY MAPPING
    # ============================================================================

    # Check if this is a string-valued property
    if is_string_valued_property(normalized):
        expected = constraint.get('value')
        if expected is None:
            return None

        # Get solver mapping for string properties
        mapping = get_solver_mapping(normalized)
        if mapping:
            method_name, params = mapping
            method = getattr(_SOLVER, method_name, None)
            if method:
                actual = method(smiles, **params)
                return {
                    "kind": "string",
                    "value": actual,
                    "expected": expected
                }
        return _UNSUPPORTED

    # Check explicit mapping first (handles most properties)
    mapping = get_solver_mapping(normalized)
    if mapping:
        method_name, params = mapping

        # Special handling for functional groups (they return dictionaries)
        if method_name == "get_functional_group_count_and_indices":
            group_name = params.get('group_name')
            if group_name:
                return {
                    "kind": "numeric",
                    "value": float(_functional_group_instances(smiles, group_name))
                }

        # Special handling for reaction templates (they return dictionaries)
        if method_name == "get_reaction_counts_and_indices":
            template_name = params.get('template_name')
            if template_name:
                # For success constraints, we need to check the success value
                if '_success' in normalized:
                    # Get the full success key
                    if not template_name.startswith('template_based_reaction_prediction_'):
                        success_key = f'template_based_reaction_prediction_{template_name}_success'
                    else:
                        success_key = template_name

                    data = _reaction_data(smiles)
                    value = data.get(success_key, 0)
                    return {
                        "kind": "numeric",
                        "value": float(value)
                    }
                else:
                    # For count constraints
                    return {
                        "kind": "numeric",
                        "value": float(_reaction_template_count(smiles, template_name))
                    }

        # Regular numeric properties
        method = getattr(_SOLVER, method_name, None)
        if method:
            try:
                result = method(smiles, **params)
                if result is not None:
                    return {
                        "kind": "numeric",
                        "value": float(result)
                    }
            except Exception as e:
                # Log error but don't crash
                print(f"Error calling {method_name} for {normalized}: {e}")
                return None

    # If no mapping was found, return unsupported
    return _UNSUPPORTED


def multi_constraint_generation_reward(
    predicted: str,
    constraints: Union[str, List[Dict]],
    *,
    return_details: bool = False,
    **kwargs
) -> Union[float, Dict[str, Any]]:
    """
    Check if generated molecule satisfies multiple constraints simultaneously.
    Uses solver.py to calculate actual property values.

    Args:
        predicted: Generated SMILES string
        constraints: List of constraint dictionaries or JSON string, each with:
            - type: constraint type (e.g., 'ring_count', 'functional_group_alcohol_count')
            - operator: constraint operator ('=', '>=', '<=', '>', '<', 'range')
            - value: exact value for '=' operator
            - min_value, max_value: for 'range' or inequality operators
            - functional_group: for functional group constraints (optional)

    Returns:
        Union[float, Dict[str, Any]]: Binary reward or a detailed report when
        ``return_details`` is ``True``.
    """
    # Handle JSON string input
    if isinstance(constraints, str):
        try:
            constraints = json.loads(constraints)
        except (json.JSONDecodeError, ValueError):
            return 0.0 if not return_details else {
                "reward": 0.0,
                "details": [],
                "supported": 0,
                "total": 0
            }

    # Handle empty constraints
    if not constraints:
        if not return_details:
            return 1.0
        # For empty constraints, still validate the SMILES
        smiles_valid = valid_smiles(predicted)
        molecule_reasonable = is_reasonable_molecule(predicted) if smiles_valid else False
        return {
            "reward": 1.0 if (smiles_valid and molecule_reasonable) else 0.0,
            "valid_smiles": smiles_valid,
            "reasonable_molecule": molecule_reasonable,
            "details": [],
            "supported": 0,
            "total": 0
        }

    normalized_smiles = _extract_smiles_prediction(predicted)

    # Basic validity checks
    smiles_valid = valid_smiles(normalized_smiles)
    molecule_reasonable = is_reasonable_molecule(normalized_smiles) if smiles_valid else False

    if not smiles_valid or not molecule_reasonable:
        if not return_details:
            return 0.0
        return {
            "reward": 0.0,
            "valid_smiles": smiles_valid,
            "reasonable_molecule": molecule_reasonable,
            "details": [],
            "supported": 0,
            "total": len(constraints)
        }

    supported_constraints = 0
    details: List[Dict[str, Any]] = []

    for constraint in constraints:
        if not isinstance(constraint, dict):
            details.append({
                "constraint": constraint,
                "supported": False,
                "satisfied": False,
                "reason": "constraint is not a dictionary"
            })
            continue

        result = _resolve_constraint_value(normalized_smiles, constraint)

        if result is _UNSUPPORTED:
            details.append({
                "constraint": constraint,
                "supported": False,
                "satisfied": False,
                "reason": "constraint type not supported"
            })
            continue

        supported_constraints += 1

        normalized_type = parse_natural_language_property(str(constraint.get('type', '')))

        entry: Dict[str, Any] = {
            "constraint": constraint,
            "supported": True,
            "normalized_type": normalized_type,
        }

        if isinstance(result, dict):
            kind = result.get("kind")
            entry["kind"] = kind

            if kind == "string":
                actual_value = result.get("value")
                expected_value = result.get("expected")
                entry["actual"] = actual_value
                entry["expected"] = expected_value
                # For molecular formulas, use normalized comparison
                if normalized_type in ['molecular_formula', 'molecular_formula_count', 'molecular_formula_value']:
                    satisfied = are_same_molecular_formula(actual_value, expected_value)
                else:
                    satisfied = actual_value == expected_value
            elif kind == "numeric":
                actual_value = result.get("value")
                entry["actual"] = actual_value
                satisfied = (
                    actual_value is not None
                    and evaluate_numeric_constraint(actual_value, constraint)
                )
            elif kind == "boolean":
                val = result.get("value")
                entry["actual"] = val
                satisfied = bool(val)
            else:
                satisfied = False
                entry["reason"] = "unsupported result kind"
        elif isinstance(result, bool):
            satisfied = result
            entry["kind"] = "boolean"
            entry["actual"] = result
        elif isinstance(result, (int, float)):
            numeric_value = float(result)
            entry["kind"] = "numeric"
            entry["actual"] = numeric_value
            satisfied = evaluate_numeric_constraint(numeric_value, constraint)
        else:
            satisfied = False
            entry["kind"] = "unknown"
            entry["reason"] = "unexpected result type"

        entry["satisfied"] = satisfied
        details.append(entry)

        if not satisfied:
            if return_details:
                return {
                    "reward": 0.0,
                    "valid_smiles": True,  # Already validated at the beginning
                    "reasonable_molecule": True,  # Already validated at the beginning
                    "details": details,
                    "supported": supported_constraints,
                    "total": len(constraints)
                }
            return 0.0

    # All constraints satisfied
    reward = 1.0 if supported_constraints > 0 else 0.0

    if return_details:
        return {
            "reward": reward,
            "valid_smiles": True,  # Always True here since we passed the validity check
            "reasonable_molecule": True,  # Always True here since we passed the validity check
            "details": details,
            "supported": supported_constraints,
            "total": len(constraints)
        }

    return reward


# Alias for simpler naming
def constraint_reward(
    predicted: str,
    constraints: Union[str, List[Dict]] = None,
    *,
    return_details: bool = False,
    **kwargs
) -> Union[float, Dict[str, Any]]:
    """
    Wrapper for constraint checking - same as multi_constraint_generation_reward.

    Args:
        predicted: Generated SMILES string
        constraints: List of constraint dictionaries or JSON string
        **kwargs: Additional arguments

    Returns:
        float: 1.0 if all constraints satisfied, 0.0 otherwise
    """
    return multi_constraint_generation_reward(
        predicted,
        constraints,
        return_details=return_details,
        **kwargs
    )
