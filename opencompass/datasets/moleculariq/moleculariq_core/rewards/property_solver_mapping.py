"""
Explicit mapping from property names to solver methods.

This module provides a clean, maintainable mapping from all property names
used in column_category_map.py to their corresponding solver methods.
This eliminates complex logic and ensures complete coverage.
"""

from typing import Dict, Tuple, Any

# Maps property names (from column_category_map.py) to (solver_method_name, parameters)
PROPERTY_TO_SOLVER_MAP: Dict[str, Tuple[str, Dict[str, Any]]] = {
    # ============================================================================
    # RING TOPOLOGY
    # ============================================================================
    "ring_count": ("get_ring_count", {}),
    "ring_index": ("get_ring_indices", {}),

    "fused_ring_count": ("get_fused_ring_count", {}),
    "fused_ring_index": ("get_fused_ring_indices", {}),

    "bridgehead_count": ("get_bridgehead_count", {}),
    "bridgehead_index": ("get_bridgehead_indices", {}),

    "spiro_count": ("get_spiro_count", {}),
    "spiro_index": ("get_spiro_indices", {}),

    # Special: smallest/largest ring size (requires parameter)
    "smallest_largest_ring_size_smallest_count": ("get_smallest_or_largest_ring_count", {"smallest": True}),
    "smallest_largest_ring_size_largest_count": ("get_smallest_or_largest_ring_count", {"smallest": False}),
    "smallest_largest_ring_size_smallest_index": ("get_smallest_or_largest_ring_indices", {"smallest": True}),
    "smallest_largest_ring_size_largest_index": ("get_smallest_or_largest_ring_indices", {"smallest": False}),

    "chain_termini_count": ("get_chain_termini_count", {}),
    "chain_termini_index": ("get_chain_termini_indices", {}),

    "branch_point_count": ("get_branch_point_count", {}),
    "branch_point_index": ("get_branch_point_indices", {}),

    "aromatic_ring_count": ("get_aromatic_ring_count", {}),
    "aromatic_ring_index": ("get_aromatic_ring_indices", {}),

    "aliphatic_ring_count": ("get_aliphatic_ring_count", {}),
    "aliphatic_ring_index": ("get_aliphatic_ring_indices", {}),

    "heterocycle_count": ("get_heterocycle_count", {}),
    "heterocycle_index": ("get_heterocycle_indices", {}),

    "saturated_ring_count": ("get_saturated_ring_count", {}),
    "saturated_ring_index": ("get_saturated_ring_indices", {}),

    # ============================================================================
    # CARBON / CHAINS
    # ============================================================================
    "csp3_carbon_count": ("get_csp3_carbon_count", {}),
    "csp3_carbon_index": ("get_csp3_carbon_indices", {}),

    "longest_carbon_chain_count": ("get_longest_carbon_chain_count", {}),
    "longest_carbon_chain_index": ("get_longest_carbon_chain_indices", {}),

    # ============================================================================
    # STEREOCHEMISTRY
    # ============================================================================

    # R/S stereochemistry (requires parameter)
    "r_s_stereocenter_r_count": ("get_r_or_s_stereocenter_count", {"r_count": True}),
    "r_s_stereocenter_s_count": ("get_r_or_s_stereocenter_count", {"r_count": False}),
    "r_s_stereocenter_r_index": ("get_r_or_s_stereocenter_indices", {"r_indices": True}),
    "r_s_stereocenter_s_index": ("get_r_or_s_stereocenter_indices", {"r_indices": False}),

    "unspecified_stereocenter_count": ("get_unspecified_stereocenter_count", {}),
    "unspecified_stereocenter_index": ("get_unspecified_stereocenter_indices", {}),

    "stereocenter_count": ("get_stereocenter_count", {}),
    "stereocenter_index": ("get_stereocenter_indices", {}),

    # E/Z stereochemistry (requires parameter)
    "e_z_stereochemistry_double_bond_e_count": ("get_e_z_stereochemistry_double_bond_count", {"e_count": True}),
    "e_z_stereochemistry_double_bond_z_count": ("get_e_z_stereochemistry_double_bond_count", {"e_count": False}),
    "e_z_stereochemistry_double_bond_e_index": ("get_e_z_stereochemistry_double_bond_indices", {"e_indices": True}),
    "e_z_stereochemistry_double_bond_z_index": ("get_e_z_stereochemistry_double_bond_indices", {"e_indices": False}),

    "stereochemistry_unspecified_double_bond_count": ("get_stereochemistry_unspecified_double_bond_count", {}),
    "stereochemistry_unspecified_double_bond_index": ("get_stereochemistry_unspecified_double_bond_indices", {}),

    # ============================================================================
    # ATOMS / COMPOSITION
    # ============================================================================
    "carbon_atom_count": ("get_carbon_atom_count", {}),
    "carbon_atom_index": ("get_carbon_atom_indices", {}),

    "hetero_atom_count": ("get_hetero_atom_count", {}),
    "hetero_atom_index": ("get_hetero_atom_indices", {}),

    "halogen_atom_count": ("get_halogen_atom_count", {}),
    "halogen_atom_index": ("get_halogen_atom_indices", {}),

    "heavy_atom_count": ("get_heavy_atom_count", {}),
    "heavy_atom_index": ("get_heavy_atom_indices", {}),

    "hydrogen_atom_count": ("get_hydrogen_count", {}),
    # Note: No hydrogen_atom_index in the mapping (hydrogen indices not typically tracked)

    # ============================================================================
    # MOLECULAR FORMULA (special - string value)
    # ============================================================================
    "molecular_formula_count": ("get_molecular_formula", {}),  # Returns string
    "molecular_formula": ("get_molecular_formula", {}),  # Alias
    "molecular_formula_value": ("get_molecular_formula", {}),  # Alias

    # ============================================================================
    # HBA/HBD/ROTATABLE BONDS
    # ============================================================================
    "hba_count": ("get_hba_count", {}),
    "hba_index": ("get_hba_indices", {}),

    "hbd_count": ("get_hbd_count", {}),
    "hbd_index": ("get_hbd_indices", {}),

    "rotatable_bond_count": ("get_rotatable_bond_count", {}),
    "rotatable_bond_index": ("get_rotatable_bond_indices", {}),

    # ============================================================================
    # OXIDATION STATES (requires element and max/min parameter)
    # ============================================================================
    # Carbon
    "oxidation_state_C_max_count": ("get_oxidation_state_count", {"element": "C", "max_oxidation": True}),
    "oxidation_state_C_min_count": ("get_oxidation_state_count", {"element": "C", "max_oxidation": False}),
    "oxidation_state_C_max_index": ("get_oxidation_state_indices", {"element": "C", "max_oxidation": True}),
    "oxidation_state_C_min_index": ("get_oxidation_state_indices", {"element": "C", "max_oxidation": False}),

    # Nitrogen
    "oxidation_state_N_max_count": ("get_oxidation_state_count", {"element": "N", "max_oxidation": True}),
    "oxidation_state_N_min_count": ("get_oxidation_state_count", {"element": "N", "max_oxidation": False}),
    "oxidation_state_N_max_index": ("get_oxidation_state_indices", {"element": "N", "max_oxidation": True}),
    "oxidation_state_N_min_index": ("get_oxidation_state_indices", {"element": "N", "max_oxidation": False}),

    # Oxygen
    "oxidation_state_O_max_count": ("get_oxidation_state_count", {"element": "O", "max_oxidation": True}),
    "oxidation_state_O_min_count": ("get_oxidation_state_count", {"element": "O", "max_oxidation": False}),
    "oxidation_state_O_max_index": ("get_oxidation_state_indices", {"element": "O", "max_oxidation": True}),
    "oxidation_state_O_min_index": ("get_oxidation_state_indices", {"element": "O", "max_oxidation": False}),

    # Phosphorus
    "oxidation_state_P_max_count": ("get_oxidation_state_count", {"element": "P", "max_oxidation": True}),
    "oxidation_state_P_min_count": ("get_oxidation_state_count", {"element": "P", "max_oxidation": False}),
    "oxidation_state_P_max_index": ("get_oxidation_state_indices", {"element": "P", "max_oxidation": True}),
    "oxidation_state_P_min_index": ("get_oxidation_state_indices", {"element": "P", "max_oxidation": False}),

    # Sulfur
    "oxidation_state_S_max_count": ("get_oxidation_state_count", {"element": "S", "max_oxidation": True}),
    "oxidation_state_S_min_count": ("get_oxidation_state_count", {"element": "S", "max_oxidation": False}),
    "oxidation_state_S_max_index": ("get_oxidation_state_indices", {"element": "S", "max_oxidation": True}),
    "oxidation_state_S_min_index": ("get_oxidation_state_indices", {"element": "S", "max_oxidation": False}),

    # ============================================================================
    # BRICS / SCAFFOLD
    # ============================================================================
    "brics_decomposition_count": ("get_brics_fragment_count", {}),
    "brics_decomposition_index": ("get_brics_bond_indices", {}),

    # Murcko scaffold (3 different methods for different purposes)
    "murcko_scaffold_count": ("get_murcko_scaffold_count", {}),  # Atom count in scaffold
    "murcko_scaffold_index": ("get_murcko_scaffold_indices", {}),  # Atom indices
    "murcko_scaffold_value": ("get_murcko_scaffold_value", {}),  # SMILES string of scaffold
}

# ============================================================================
# SPECIAL HANDLING REQUIRED
# ============================================================================

# Functional groups: These are handled via get_functional_group_count_and_indices()
# Pattern: functional_group_{group_name}_{suffix}
# where suffix is one of: count, index, nbrInstances
# There are 135+ functional groups defined in FG_KEYS

# Reaction templates: These are handled via get_reaction_counts_and_indices()
# Pattern: template_based_reaction_prediction_{template_name}_success
# There are 58 reaction templates

def get_functional_group_mapping(group_name: str, suffix: str) -> Tuple[str, Dict[str, Any]]:
    """
    Get solver method and parameters for functional group properties.

    Args:
        group_name: Name of the functional group (e.g., 'aldehyde', 'ketone')
        suffix: Type of property ('count', 'index', 'nbrInstances')

    Returns:
        Tuple of (method_name, parameters)
    """
    # All functional groups use the same method
    return ("get_functional_group_count_and_indices", {"group_name": group_name})


def get_reaction_template_mapping(template_name: str) -> Tuple[str, Dict[str, Any]]:
    """
    Get solver method and parameters for reaction template properties.

    Args:
        template_name: Name of the reaction template

    Returns:
        Tuple of (method_name, parameters)
    """
    # All reaction templates use the same method
    return ("get_reaction_counts_and_indices", {"template_name": template_name})


def is_string_valued_property(property_name: str) -> bool:
    """
    Check if a property returns a string value rather than numeric.

    Args:
        property_name: Normalized property name

    Returns:
        True if property returns string, False if numeric
    """
    string_properties = {
        'molecular_formula', 'molecular_formula_count', 'molecular_formula_value',
        'murcko_scaffold_value'
    }
    return property_name in string_properties


def get_solver_mapping(property_name: str) -> Tuple[str, Dict[str, Any]]:
    """
    Get the solver method and parameters for any property.

    Args:
        property_name: Normalized property name

    Returns:
        Tuple of (method_name, parameters) or None if not found
    """
    # Check explicit mapping first
    if property_name in PROPERTY_TO_SOLVER_MAP:
        return PROPERTY_TO_SOLVER_MAP[property_name]

    # Special handling for oxidation state properties (preserve element case)
    if property_name.startswith('oxidation_state_'):
        parts = property_name.split('_')
        if len(parts) >= 4:  # oxidation_state_X_min/max_count/index
            # Reconstruct with uppercase element symbol
            element = parts[2].upper()  # Convert element back to uppercase
            min_max = parts[3]
            suffix = parts[4] if len(parts) > 4 else ''
            if suffix:
                reconstructed = f'oxidation_state_{element}_{min_max}_{suffix}'
            else:
                reconstructed = f'oxidation_state_{element}_{min_max}'
            if reconstructed in PROPERTY_TO_SOLVER_MAP:
                return PROPERTY_TO_SOLVER_MAP[reconstructed]

    # Check functional groups
    if property_name.startswith('functional_group_'):
        parts = property_name.split('_')
        if len(parts) >= 3:
            # Extract group name and suffix
            # Format: functional_group_{group_name}_{suffix}
            suffix = parts[-1]
            group_name = '_'.join(parts[2:-1])
            return get_functional_group_mapping(group_name, suffix)

    # Check reaction templates
    if property_name.startswith('template_based_reaction_prediction_'):
        template = property_name[len('template_based_reaction_prediction_'):]
        # Remove suffix if present
        for suffix in ['_count', '_index', '_success', '_products']:
            if template.endswith(suffix):
                template = template[:-len(suffix)]
                break
        return get_reaction_template_mapping(template)

    # Legacy reaction format
    if property_name.startswith('reaction_'):
        template = property_name[len('reaction_'):]
        if template.endswith('_count'):
            template = template[:-6]
        return get_reaction_template_mapping(template)

    return None