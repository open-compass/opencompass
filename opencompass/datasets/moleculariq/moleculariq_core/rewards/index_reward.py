"""
Reward functions for index identification tasks (single and multi).
Ground truth is provided in the target dictionary - no recalculation.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union

from ..properties import canonicalize_property_name


def parse_indices_string(s: str) -> Optional[List[int]]:
    """
    Parse comma-separated indices like '0,2,3' or '[0,2,3]'.

    Args:
        s: String containing indices

    Returns:
        List of integer indices, or None if invalid format
    """
    if not s or s.lower() in ['none', 'empty', '[]', '()', '{}', '']:
        return []

    # Remove brackets
    s = s.strip()
    if not s or s.lower() in ['none', 'empty', '[]', '()', '{}', '']:
        return []

    s = re.sub(r'^\[|\]$|^\(|\)$|^\{|\}$', '', s.strip())

    # After removing brackets, check if empty
    if not s:
        return []

    try:
        # Split and validate ALL parts are numeric
        parts = [x.strip() for x in s.split(',') if x.strip()]

        if not parts:
            return []

        indices = []
        for part in parts:
            # Check if it's a valid integer
            idx = int(part)
            if idx < 0:
                return None  # Invalid negative index
            indices.append(idx)

        return sorted(set(indices))  # Remove duplicates and sort
    except (ValueError, TypeError):
        return None


def _normalize_index_dict(raw: Dict[str, Any]) -> Optional[Dict[str, List[int]]]:
    """Normalize index dictionary keys.

    Returns None if different raw keys normalize to the same canonical key
    (duplicate detection).
    """
    from .._nlp.mappings import parse_natural_language

    normalized: Dict[str, List[int]] = {}
    seen_raw_keys: Dict[str, str] = {}  # normalized_key -> original raw key

    for key, indices in raw.items():
        normalized_key = canonicalize_property_name(parse_natural_language(key.lower()))

        # Check for duplicate normalized keys from different raw keys
        if normalized_key in seen_raw_keys and seen_raw_keys[normalized_key] != key:
            return None  # Different raw keys normalize to same key
        seen_raw_keys[normalized_key] = key

        if isinstance(indices, str):
            parsed = parse_indices_string(indices)
            if parsed is None:
                continue
            normalized[normalized_key] = parsed
        elif isinstance(indices, list):
            cleaned = []
            for idx in indices:
                try:
                    if isinstance(idx, (int, float)) and idx >= 0:
                        cleaned.append(int(idx))
                except (ValueError, TypeError):
                    continue
            normalized[normalized_key] = sorted(set(cleaned))
    return normalized


def multi_index_identification_reward(
    predicted: Union[str, List, Dict],
    target: Union[str, Dict],
    *,
    return_details: bool = False
) -> Union[float, Dict[str, Any]]:
    """
    Reward for both single-index and multi-index identification tasks.
    Target dict IS the ground truth - we don't recalculate.

    For single index tasks (target has 1 key), accepts:
    - List: [0, 2, 3]
    - String: "0,2,3" or "[0,2,3]"
    - Dict with one key: {"aromatic_ring_index": [0,2,3]}

    For multi-index tasks:
    - Dict format: {"type1": [indices], "type2": [indices]}
    - JSON string of dict

    Args:
        predicted: Predicted atom indices (list, dict, or string)
        target: Target atom indices by type (dict or string)

    Returns:
        Union[float, Dict[str, Any]]: Either the traditional score (float) or a
        detail dictionary when ``return_details`` is ``True``.
    """
    # Parse target to dictionary
    if isinstance(target, str):
        try:
            target = json.loads(target)
        except (json.JSONDecodeError, ValueError):
            target_dict = {}
            for item in target.split(';'):
                if ':' in item:
                    prop, indices = item.strip().split(':', 1)
                    parsed = parse_indices_string(indices)
                    if parsed is not None:
                        target_dict[prop.strip()] = parsed
            target = target_dict

    if not isinstance(target, dict):
        return 0.0 if not return_details else {
            "reward": 0.0,
            "details": {},
            "matched": 0,
            "total": 0,
            "extra_predictions": {}
        }

    normalized_target_full = _normalize_index_dict(target)

    # Target normalization failed (e.g., duplicates) - shouldn't happen but handle gracefully
    if normalized_target_full is None:
        return 0.0 if not return_details else {
            "reward": 0.0,
            "details": {},
            "matched": 0,
            "total": 0,
            "extra_predictions": {}
        }

    # Check if single index task (target has exactly one key)
    if len(normalized_target_full) == 1:
        target_key, target_indices_list = next(iter(normalized_target_full.items()))
        target_set = set(target_indices_list)

        # Handle single index predicted formats
        pred_indices = None

        if isinstance(predicted, list):
            # Direct list
            pred_indices = predicted

        elif isinstance(predicted, str):
            # Could be: "0,2,3", "[0,2,3]", or '{"type": [0,2,3]}'
            predicted = predicted.strip()

            # Try parsing as indices string
            parsed = parse_indices_string(predicted)
            if parsed is not None:
                pred_indices = parsed
            else:
                # Try parsing as JSON
                try:
                    pred_data = json.loads(predicted)
                    if isinstance(pred_data, list):
                        pred_indices = pred_data
                    elif isinstance(pred_data, dict) and len(pred_data) == 1:
                        # Get the single value
                        pred_indices = list(pred_data.values())[0]
                except (json.JSONDecodeError, ValueError):
                    # Try semicolon format
                    if ':' in predicted:
                        parts = predicted.split(':', 1)
                        if len(parts) == 2:
                            parsed = parse_indices_string(parts[1])
                            if parsed is not None:
                                pred_indices = parsed

        elif isinstance(predicted, dict) and len(predicted) == 1:
            # Single-key dict
            pred_indices = list(predicted.values())[0]

        # Validate and compare
        if pred_indices is None:
            return 0.0 if not return_details else {
                "reward": 0.0,
                "details": {
                    target_key: {
                        "target": sorted(target_set),
                        "predicted": None,
                        "match": False
                    }
                },
                "matched": 0,
                "total": 1,
                "extra_predictions": {}
            }

        if isinstance(pred_indices, str):
            pred_indices = parse_indices_string(pred_indices)

        if not isinstance(pred_indices, list):
            return 0.0 if not return_details else {
                "reward": 0.0,
                "details": {
                    target_key: {
                        "target": sorted(target_set),
                        "predicted": None,
                        "match": False
                    }
                },
                "matched": 0,
                "total": 1,
                "extra_predictions": {}
            }

        # Filter out invalid indices
        pred_set = set()
        for idx in pred_indices:
            try:
                if isinstance(idx, (int, float)) and idx >= 0:
                    pred_set.add(int(idx))
            except (ValueError, TypeError):
                continue

        match = pred_set == target_set
        reward = 1.0 if match else 0.0

        if return_details:
            extra_predictions = {}
            if isinstance(predicted, dict) and len(predicted) > 1:
                normalized = _normalize_index_dict(predicted)
                if normalized is not None:
                    for key, value in normalized.items():
                        if key != target_key:
                            extra_predictions[key] = value

            return {
                "reward": reward,
                "details": {
                    target_key: {
                        "target": sorted(target_set),
                        "predicted": sorted(pred_set),
                        "match": match
                    }
                },
                "matched": 1 if match else 0,
                "total": 1,
                "extra_predictions": extra_predictions
            }

        return reward

    # Multi-index task handling
    pred_dict = None

    if isinstance(predicted, dict):
        pred_dict = predicted
    elif isinstance(predicted, str):
        # Try JSON parsing
        try:
            pred_dict = json.loads(predicted)
            if not isinstance(pred_dict, dict):
                return 0.0
        except (json.JSONDecodeError, ValueError):
            # Try semicolon-separated format
            pred_dict = {}
            for item in predicted.split(';'):
                if ':' in item:
                    prop, indices = item.strip().split(':', 1)
                    parsed = parse_indices_string(indices)
                    if parsed is not None:
                        pred_dict[prop.strip()] = parsed

    if not pred_dict:
        if return_details:
            details = {
                key: {
                    "target": sorted(set(indices)) if isinstance(indices, list) else [],
                    "predicted": None,
                    "match": False
                }
                for key, indices in target.items()
            }
            return {
                "reward": 0.0,
                "details": details,
                "matched": 0,
                "total": len(details),
                "extra_predictions": {}
            }
        return 0.0

    norm_target = {
        key: set(value)
        for key, value in normalized_target_full.items()
    }

    # Normalize prediction - check for duplicate keys
    normalized_pred = _normalize_index_dict(pred_dict)
    if normalized_pred is None:
        # Duplicate keys detected in prediction - return 0
        if return_details:
            details = {
                key: {
                    "target": sorted(set(indices)) if isinstance(indices, list) else [],
                    "predicted": None,
                    "match": False
                }
                for key, indices in target.items()
            }
            return {
                "reward": 0.0,
                "details": details,
                "matched": 0,
                "total": len(details),
                "extra_predictions": {}
            }
        return 0.0

    norm_pred = {
        key: set(value)
        for key, value in normalized_pred.items()
    }

    if not norm_target:
        if return_details:
            return {
                "reward": 0.0,
                "details": {},
                "matched": 0,
                "total": 0,
                "extra_predictions": {
                    key: sorted(list(value)) for key, value in norm_pred.items()
                }
            }
        return 0.0

    matched = 0
    details: Dict[str, Dict[str, Any]] = {}

    for key, target_set in norm_target.items():
        pred_set = norm_pred.get(key)
        match = pred_set is not None and pred_set == target_set
        if match:
            matched += 1
        details[key] = {
            "target": sorted(list(target_set)),
            "predicted": sorted(list(pred_set)) if pred_set is not None else None,
            "match": match
        }

    reward = 1.0 if matched == len(norm_target) else 0.0

    if return_details:
        extra_predictions = {
            key: sorted(list(value))
            for key, value in norm_pred.items()
            if key not in norm_target
        }
        return {
            "reward": reward,
            "details": details,
            "matched": matched,
            "total": len(norm_target),
            "extra_predictions": extra_predictions
        }

    return reward


# Alias for backward compatibility
def single_index_reward(
    predicted: Union[str, List, Dict],
    target: Dict,
    *,
    return_details: bool = False
) -> Union[float, Dict[str, Any]]:
    """
    Wrapper for single index tasks.
    Extracts the single list from target dict and compares.

    Args:
        predicted: Predicted index list
        target: Dictionary with single property-indices pair

    Returns:
        float: 1.0 if match, 0.0 otherwise
    """
    return multi_index_identification_reward(
        predicted,
        target,
        return_details=return_details
    )
