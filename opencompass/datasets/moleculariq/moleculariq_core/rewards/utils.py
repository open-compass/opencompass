"""
Shared utilities for reward functions.
"""

import re
from typing import Dict, Any

from ..properties import canonicalize_property_name
from rdkit import Chem
from rdkit.Chem import SanitizeMol
from .._nlp.mappings import parse_natural_language

def valid_smiles(smiles: str) -> bool:
    """
    Check if SMILES string is valid using RDKit.

    Args:
        smiles: SMILES string to validate

    Returns:
        bool: True if valid SMILES, False otherwise
    """
    if not smiles or not isinstance(smiles, str):
        return False

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        SanitizeMol(mol)
        return True
    except:
        return False


def is_reasonable_molecule(smiles: str) -> bool:
    """
    Check if a SMILES string represents a reasonable molecule.
    Simplified version: only checks if RDKit can parse and sanitize.

    Args:
        smiles: SMILES string to check

    Returns:
        bool: True if reasonable molecule, False otherwise
    """
    return valid_smiles(smiles)


def evaluate_numeric_constraint(actual_value: float, constraint: Dict[str, Any]) -> bool:
    """
    Evaluate if a numeric value satisfies a constraint.

    Args:
        actual_value: The actual value to check
        constraint: Dict with 'operator' and value fields
            - operator: '=', '>', '<', '>=', '<=', 'range'
            - value: target value for '=' operator
            - min_value, max_value: for range or inequality operators

    Returns:
        bool: True if constraint satisfied, False otherwise
    """
    operator = constraint.get('operator', '=')

    if operator == '=':
        target = constraint.get('value')
        if target is None:
            return False
        # For integers, use exact comparison
        if isinstance(actual_value, int) and isinstance(target, int):
            return actual_value == target
        # For floats, use tolerance
        return abs(actual_value - target) < 1e-6

    elif operator == '>=':
        min_val = constraint.get('min_value', constraint.get('value'))
        if min_val is None:
            return False
        return actual_value >= min_val

    elif operator == '<=':
        max_val = constraint.get('max_value', constraint.get('value'))
        if max_val is None:
            return False
        return actual_value <= max_val

    elif operator == '>':
        min_val = constraint.get('min_value', constraint.get('value'))
        if min_val is None:
            return False
        return actual_value > min_val

    elif operator == '<':
        max_val = constraint.get('max_value', constraint.get('value'))
        if max_val is None:
            return False
        return actual_value < max_val

    elif operator in ['range', '-']:
        min_val = constraint.get('min_value', float('-inf'))
        max_val = constraint.get('max_value', float('inf'))
        return min_val <= actual_value <= max_val

    else:
        # Unknown operator
        return False


def parse_natural_language_property(name: str) -> str:
    """
    Convert natural language property name to technical key.
    Uses the natural language mappings from src/natural_language/mappings.py

    Args:
        name: Natural language property name

    Returns:
        str: Technical property key
    """
    try:
        technical = parse_natural_language(name)
    except ImportError:
        technical = name.lower().replace(' ', '_')

    return canonicalize_property_name(technical)


def normalize_molecular_formula(formula: str) -> str:
    """
    Normalize a molecular formula by handling various formats.
    Converts subscripts, handles case variations, etc.

    Args:
        formula: Raw molecular formula string

    Returns:
        Normalized formula string
    """
    if not formula or not isinstance(formula, str):
        return ""

    # Translation tables for unicode subscripts/superscripts to normal
    sub2normal = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

    # Normalize the formula
    normalized = formula.translate(sub2normal)

    # Remove common formatting characters
    normalized = normalized.replace(" ", "")  # Remove spaces
    normalized = normalized.replace("-", "")  # Remove hyphens
    normalized = normalized.replace("·", "")  # Remove dots (for hydrates)

    return normalized


def parse_molecular_formula(formula: str) -> Dict[str, int]:
    """
    Parse a molecular formula into element counts.
    Handles various formats including subscripts and case variations.

    Args:
        formula: Molecular formula string (e.g., "C2H6O", "H6C2O", "c2h6o", "C₂H₆O")

    Returns:
        Dict mapping elements to their counts
    """
    if not formula or not isinstance(formula, str):
        return {}

    # Normalize the formula first
    formula = normalize_molecular_formula(formula)

    # Pattern to match element symbols (case-insensitive) and optional counts
    # This handles both proper case (Ca) and lowercase (ca, CA)
    pattern = r'([A-Z][a-z]?|[a-z]+)(\d*)'

    elements = {}
    matches = re.findall(pattern, formula, re.IGNORECASE)

    for element, count in matches:
        # Normalize element symbol to proper case
        if len(element) == 1:
            element = element.upper()
        elif len(element) == 2:
            element = element[0].upper() + element[1].lower()
        else:
            # Handle cases like "ca" or "CA" for Calcium
            element = element[0].upper() + element[1:].lower()

        count = int(count) if count else 1
        if element in elements:
            elements[element] += count
        else:
            elements[element] = count

    return elements


def are_same_molecular_formula(formula1: str, formula2: str) -> bool:
    """
    Check if two molecular formulas represent the same molecule.
    Handles different orderings (e.g., "C2H6O" vs "H6C2O").

    Args:
        formula1: First molecular formula
        formula2: Second molecular formula

    Returns:
        True if formulas represent the same molecule
    """
    try:
        elements1 = parse_molecular_formula(formula1)
        elements2 = parse_molecular_formula(formula2)

        if set(elements1.keys()) != set(elements2.keys()):
            return False

        return all(elements1[element] == elements2[element] for element in elements1)
    except:
        # If parsing fails, fall back to string comparison
        return formula1 == formula2
