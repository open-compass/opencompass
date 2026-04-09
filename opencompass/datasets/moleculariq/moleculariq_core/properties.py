from typing import Dict


INDEX_MAP = {
    # Ring topology
    "ring": ["ring_index"],
    "fused_ring": ["fused_ring_index"],
    "bridgehead": ["bridgehead_index"],
    "smallest_largest_ring_size": [
        "smallest_largest_ring_size_smallest_index",
        "smallest_largest_ring_size_largest_index",
    ],
    "chain_termini": ["chain_termini_index"],
    "branch_point": ["branch_point_index"],
    "aromatic_ring": ["aromatic_ring_index"],
    "aliphatic_ring": ["aliphatic_ring_index"],
    "heterocycle": ["heterocycle_index"],
    "saturated_ring": ["saturated_ring_index"],

    # Carbon / chains
    "csp3_carbon": ["csp3_carbon_index"],
    "longest_carbon_chain": ["longest_carbon_chain_index"],

    # Stereochemistry (exception: grouped)
    "r_s_stereocenter": [
        "r_s_stereocenter_r_index",
        "r_s_stereocenter_s_index"
    ],
    "unspecified_stereocenter": [
        "unspecified_stereocenter_index"
        ],
    "stereocenter": [
        "stereocenter_index"
        ],
    
    "e_z_stereochemistry_double_bond": [
        "e_z_stereochemistry_double_bond_e_index",
        "e_z_stereochemistry_double_bond_z_index"
        ],
    "stereochemistry_unspecified_double_bond": [
        "stereochemistry_unspecified_double_bond_index"
        ],

    # Atoms / composition
    "carbon_atom": [
        "carbon_atom_index"
        ],
    "hetero_atom": [
        "hetero_atom_index"
        ],
    "halogen_atom": [
        "halogen_atom_index"
        ],
    "heavy_atom": [
        "heavy_atom_index"
    ],

    "hba": [
        "hba_index"
    ],
    "hbd": [
        "hbd_index"
    ],
    "rotatable_bond": [
        "rotatable_bond_index"
    ],

    # Oxidation state (exception: grouped)
    "oxidation_state": [
        "oxidation_state_C_max_index", "oxidation_state_C_min_index",
        "oxidation_state_N_max_index", "oxidation_state_N_min_index",
        "oxidation_state_O_max_index", "oxidation_state_O_min_index",
        "oxidation_state_P_max_index", "oxidation_state_P_min_index",
        "oxidation_state_S_max_index", "oxidation_state_S_min_index",
    ],

    # BRICS / scaffold
    "brics_decomposition": ["brics_decomposition_index"],
    "murcko_scaffold": ["murcko_scaffold_index"],

    # Functional groups
    "functional_groups": [
        
    ],

}


KEY_ALIAS_MAP: Dict[str, str] = {
    # Ring size composites
    "smallest_largest_ring_size_smallest_count": "smallest_ring_atom_count",
    "smallest_largest_ring_size_largest_count": "largest_ring_atom_count",
    "smallest_largest_ring_size_smallest_index": "smallest_ring_atom_indices",
    "smallest_largest_ring_size_largest_index": "largest_ring_atom_indices",

    # Chain termini / longest chain
    "chain_termini_count": "terminal_carbon_count",
    "chain_termini_index": "terminal_carbon_indices",
    "longest_carbon_chain_count": "longest_chain_atom_count",
    "longest_carbon_chain_index": "longest_chain_atom_indices",

    # Stereochemistry
    "r_s_stereocenter_r_count": "r_stereocenter_count",
    "r_s_stereocenter_s_count": "s_stereocenter_count",
    "r_s_stereocenter_r_index": "r_stereocenter_indices",
    "r_s_stereocenter_s_index": "s_stereocenter_indices",
    "unspecified_stereocenter_count": "unspecified_chiral_center_count",
    "unspecified_stereocenter_index": "unspecified_chiral_center_indices",
    "stereocenter_count": "chiral_center_count",
    "stereocenter_index": "chiral_center_indices",
    "stereochemistry_unspecified_double_bond_count": "unspecified_double_bond_count",
    "stereochemistry_unspecified_double_bond_index": "unspecified_double_bond_indices",

    # Oxidation state (upper- and lowercase variants)
    "oxidation_state_C_max_count": "carbon_max_oxidation_count",
    "oxidation_state_C_min_count": "carbon_min_oxidation_count",
    "oxidation_state_c_max_count": "carbon_max_oxidation_count",
    "oxidation_state_c_min_count": "carbon_min_oxidation_count",
    "oxidation_state_N_max_count": "nitrogen_max_oxidation_count",
    "oxidation_state_N_min_count": "nitrogen_min_oxidation_count",
    "oxidation_state_n_max_count": "nitrogen_max_oxidation_count",
    "oxidation_state_n_min_count": "nitrogen_min_oxidation_count",
    "oxidation_state_O_max_count": "oxygen_max_oxidation_count",
    "oxidation_state_O_min_count": "oxygen_min_oxidation_count",
    "oxidation_state_o_max_count": "oxygen_max_oxidation_count",
    "oxidation_state_o_min_count": "oxygen_min_oxidation_count",
    "oxidation_state_P_max_count": "phosphorus_max_oxidation_count",
    "oxidation_state_P_min_count": "phosphorus_min_oxidation_count",
    "oxidation_state_p_max_count": "phosphorus_max_oxidation_count",
    "oxidation_state_p_min_count": "phosphorus_min_oxidation_count",
    "oxidation_state_S_max_count": "sulfur_max_oxidation_count",
    "oxidation_state_S_min_count": "sulfur_min_oxidation_count",
    "oxidation_state_s_max_count": "sulfur_max_oxidation_count",
    "oxidation_state_s_min_count": "sulfur_min_oxidation_count",
    "oxidation_state_C_max_index": "carbon_max_oxidation_indices",
    "oxidation_state_C_min_index": "carbon_min_oxidation_indices",
    "oxidation_state_c_max_index": "carbon_max_oxidation_indices",
    "oxidation_state_c_min_index": "carbon_min_oxidation_indices",
    "oxidation_state_N_max_index": "nitrogen_max_oxidation_indices",
    "oxidation_state_N_min_index": "nitrogen_min_oxidation_indices",
    "oxidation_state_n_max_index": "nitrogen_max_oxidation_indices",
    "oxidation_state_n_min_index": "nitrogen_min_oxidation_indices",
    "oxidation_state_O_max_index": "oxygen_max_oxidation_indices",
    "oxidation_state_O_min_index": "oxygen_min_oxidation_indices",
    "oxidation_state_o_max_index": "oxygen_max_oxidation_indices",
    "oxidation_state_o_min_index": "oxygen_min_oxidation_indices",
    "oxidation_state_P_max_index": "phosphorus_max_oxidation_indices",
    "oxidation_state_P_min_index": "phosphorus_min_oxidation_indices",
    "oxidation_state_p_max_index": "phosphorus_max_oxidation_indices",
    "oxidation_state_p_min_index": "phosphorus_min_oxidation_indices",
    "oxidation_state_S_max_index": "sulfur_max_oxidation_indices",
    "oxidation_state_S_min_index": "sulfur_min_oxidation_indices",
    "oxidation_state_s_max_index": "sulfur_max_oxidation_indices",
    "oxidation_state_s_min_index": "sulfur_min_oxidation_indices",

    # Miscellaneous clarifications
    "brics_decomposition_count": "brics_fragment_count",
    "brics_decomposition_index": "brics_fragment_indices",
    "murcko_scaffold_count": "murcko_scaffold_atom_count",
    "murcko_scaffold_index": "murcko_scaffold_atom_indices",
}

COUNT_MAP = {
    # Ring topology
    "ring": ["ring_count"],
    "fused_ring": ["fused_ring_count"],
    "bridgehead": ["bridgehead_count"],
    "smallest_largest_ring_size": [
        "smallest_largest_ring_size_smallest_count",
        "smallest_largest_ring_size_largest_count",
    ],
    "chain_termini": ["chain_termini_count"],
    "branch_point": ["branch_point_count"],
    "aromatic_ring": ["aromatic_ring_count"],
    "aliphatic_ring": ["aliphatic_ring_count"],
    "heterocycle": ["heterocycle_count"],
    "saturated_ring": ["saturated_ring_count"],

    # Carbon / chains
    "csp3_carbon": ["csp3_carbon_count"],
    "longest_carbon_chain": ["longest_carbon_chain_count"],

    # Stereochemistry (exception: grouped)
    "r_s_stereocenter": [
        "r_s_stereocenter_r_count",
        "r_s_stereocenter_s_count"  
    ],
    "unspecified_stereocenter": ["unspecified_stereocenter_count"],
    "stereocenter": ["stereocenter_count"],

    "e_z_stereochemistry_double_bond": [
        "e_z_stereochemistry_double_bond_e_count",
        "e_z_stereochemistry_double_bond_z_count"
    ],
    "stereochemistry_unspecified_double_bond": ["stereochemistry_unspecified_double_bond_count"],

    # Atoms / composition
    "carbon_atom": ["carbon_atom_count"],
    "hetero_atom": ["hetero_atom_count"],
    "halogen_atom": ["halogen_atom_count"],
    "heavy_atom": ["heavy_atom_count"],
    "hydrogen_atom": ["hydrogen_atom_count"],

    # Formula
    "molecular_formula": ["molecular_formula_count"],

    # HBA/HBD/rotatable (functional-group-ish)
    "hba": [
        "hba_count",
    ],
    "hbd": [
        "hbd_count",
    ],
    "rotatable_bond": [
        "rotatable_bond_count",
    ],

    # Oxidation state (exception: grouped)
    "oxidation_state": [
        "oxidation_state_C_max_count", "oxidation_state_C_min_count",
        "oxidation_state_N_max_count", "oxidation_state_N_min_count",
        "oxidation_state_O_max_count", "oxidation_state_O_min_count",
        "oxidation_state_P_max_count", "oxidation_state_P_min_count",
        "oxidation_state_S_max_count", "oxidation_state_S_min_count",
    ],

    # BRICS / scaffold
    "brics_decomposition": ["brics_decomposition_count"],
    "murcko_scaffold": ["murcko_scaffold_count"],

    # Functional groups
    "functional_groups": [
    ],
}

CONSTRAINT_MAP = {
    # Ring topology
    "ring": ["ring_count"],
    "fused_ring": ["fused_ring_count"],
    "bridgehead": ["bridgehead_count"],
    "smallest_largest_ring_size": [
        "smallest_largest_ring_size_smallest_count",
        "smallest_largest_ring_size_largest_count",
    ],
    "chain_termini": ["chain_termini_count"],
    "branch_point": ["branch_point_count"],
    "aromatic_ring": ["aromatic_ring_count"],
    "aliphatic_ring": ["aliphatic_ring_count"],
    "heterocycle": ["heterocycle_count"],
    "saturated_ring": ["saturated_ring_count"],

    # Carbon / chains
    "csp3_carbon": ["csp3_carbon_count"],
    "longest_carbon_chain": ["longest_carbon_chain_count"],

    # Stereochemistry (exception: grouped)
    "r_s_stereocenter": [
        "r_s_stereocenter_r_count",
        "r_s_stereocenter_s_count"  
    ],
    "unspecified_stereocenter": ["unspecified_stereocenter_count"],
    "stereocenter": ["stereocenter_count"],

    "e_z_stereochemistry_double_bond": [
        "e_z_stereochemistry_double_bond_e_count",
        "e_z_stereochemistry_double_bond_z_count"
    ],
    "stereochemistry_unspecified_double_bond": ["stereochemistry_unspecified_double_bond_count"],

    # Atoms / composition
    "carbon_atom": ["carbon_atom_count"],
    "hetero_atom": ["hetero_atom_count"],
    "halogen_atom": ["halogen_atom_count"],
    "heavy_atom": ["heavy_atom_count"],
    "hydrogen_atom": ["hydrogen_atom_count"],

    # Formula
    "molecular_formula": ["molecular_formula_count"],

    # HBA/HBD/rotatable (functional-group-ish)
    "hba": [
        "hba_count",
    ],
    "hbd": [
        "hbd_count",
    ],
    "rotatable_bond": [
        "rotatable_bond_count",
    ],

    # Oxidation state (exception: grouped)
    "oxidation_state": [
        "oxidation_state_C_max_count", "oxidation_state_C_min_count",
        "oxidation_state_N_max_count", "oxidation_state_N_min_count",
        "oxidation_state_O_max_count", "oxidation_state_O_min_count",
        "oxidation_state_P_max_count", "oxidation_state_P_min_count",
        "oxidation_state_S_max_count", "oxidation_state_S_min_count",
    ],

    # BRICS / scaffold
    "brics_decomposition": ["brics_decomposition_count"],

    "murcko_scaffold": ["murcko_scaffold_count", "murcko_scaffold_value"],

    # Functional groups
    "functional_groups": [
    ],

    # reaction_success
    "reaction_success": [
        "template_based_reaction_prediction_primary_alcohol_to_aldehyde_success",
        "template_based_reaction_prediction_secondary_alcohol_to_ketone_success",
        "template_based_reaction_prediction_aldehyde_to_carboxylic_acid_success",
        "template_based_reaction_prediction_ketone_to_secondary_alcohol_success",
        "template_based_reaction_prediction_aldehyde_to_primary_alcohol_success",
        "template_based_reaction_prediction_carboxylic_acid_to_primary_alcohol_success",
        "template_based_reaction_prediction_nitro_to_amine_success",
        "template_based_reaction_prediction_imine_to_amine_success",
        "template_based_reaction_prediction_alkene_to_alkane_success",
        "template_based_reaction_prediction_alkyne_to_alkene_success",
        "template_based_reaction_prediction_alcohol_to_alkyl_halide_Cl_success",
        "template_based_reaction_prediction_alcohol_to_alkyl_halide_Br_success",
        "template_based_reaction_prediction_alkyl_halide_to_alcohol_success",
        "template_based_reaction_prediction_alkyl_halide_to_azide_success",
        "template_based_reaction_prediction_alkyl_halide_to_nitrile_success",
        "template_based_reaction_prediction_ester_hydrolysis_to_acid_success",
        "template_based_reaction_prediction_amide_hydrolysis_to_acid_success",
        "template_based_reaction_prediction_nitrile_hydrolysis_success",
        "template_based_reaction_prediction_hydration_of_alkene_success",
        "template_based_reaction_prediction_hydrohalogenation_HCl_success",
        "template_based_reaction_prediction_hydrohalogenation_HBr_success",
        "template_based_reaction_prediction_epoxidation_success",
        "template_based_reaction_prediction_dehydration_alcohol_success",
        "template_based_reaction_prediction_dehydrohalogenation_success",
        "template_based_reaction_prediction_nitration_success",
        "template_based_reaction_prediction_bromination_success",
        "template_based_reaction_prediction_chlorination_success",
        "template_based_reaction_prediction_beckmann_rearrangement_success",
        "template_based_reaction_prediction_decarboxylation_success",
        "template_based_reaction_prediction_boc_protection_success",
        "template_based_reaction_prediction_boc_deprotection_success",
        "template_based_reaction_prediction_acetylation_success",
        "template_based_reaction_prediction_deacetylation_success",
        "template_based_reaction_prediction_oxime_formation_success",
        "template_based_reaction_prediction_hydrazone_formation_success",
        "template_based_reaction_prediction_benzylic_oxidation_success",
        "template_based_reaction_prediction_benzylic_alcohol_oxidation_success",
        "template_based_reaction_prediction_sulfide_to_sulfoxide_success",
        "template_based_reaction_prediction_azide_to_amine_success",
        "template_based_reaction_prediction_ester_to_alcohol_success",
        "template_based_reaction_prediction_nitrile_to_amine_success",
        "template_based_reaction_prediction_amide_to_amine_success",
        "template_based_reaction_prediction_alcohol_to_alkyl_halide_I_success",
        "template_based_reaction_prediction_alcohol_to_tosylate_success",
        "template_based_reaction_prediction_alcohol_to_mesylate_success",
        "template_based_reaction_prediction_dibromination_success",
        "template_based_reaction_prediction_dichlorination_success",
        "template_based_reaction_prediction_aromatic_sulfonation_success",
        "template_based_reaction_prediction_aromatic_iodination_success",
        "template_based_reaction_prediction_aromatic_fluorination_success",
        "template_based_reaction_prediction_acid_to_acid_chloride_success",
        "template_based_reaction_prediction_acid_chloride_to_ester_methyl_success",
        "template_based_reaction_prediction_acid_chloride_to_amide_success",
        "template_based_reaction_prediction_aromatic_amine_to_diazonium_success",
        "template_based_reaction_prediction_epoxide_to_diol_success",
        "template_based_reaction_prediction_epoxide_to_halohydrin_Br_success",
        "template_based_reaction_prediction_epoxide_to_halohydrin_Cl_success",
        "template_based_reaction_prediction_ozonolysis_terminal_success",
    ],
}

FG_KEYS = [
    "aldehyde","ketone","carboxylic_acid","ester","amide","lactone","lactam",
    "acyl_chloride","acyl_bromide","acyl_fluoride","acyl_iodide","anhydride",
    "primary_amine","secondary_amine","tertiary_amine","quaternary_ammonium",
    "imine","nitrile","nitro","nitroso","azide","hydrazine","hydrazone","oxime",
    "isocyanate","isothiocyanate","carbodiimide","urea","thiourea","guanidine",
    "alcohol","phenol","ether","aryl_ether","epoxide","peroxide","hydroperoxide",
    "enol","hemiacetal","hemiketal","acetal","ketal","thiol","thioether",
    "disulfide","sulfoxide","sulfone","sulfonic_acid","sulfonamide",
    "sulfonic_ester","thioketone","thioaldehyde",
    "alkyl_fluoride","alkyl_chloride","alkyl_bromide","alkyl_iodide",
    "aryl_fluoride","aryl_chloride","aryl_bromide","aryl_iodide",
    "vinyl_halide","trifluoromethyl","perfluoroalkyl",
    "phosphine","phosphine_oxide","phosphonium","phosphonic_acid","phosphate",
    "phosphonate","phosphoramide",
    "alkene","alkyne","allene","aromatic","conjugated_system","enamine",
    "enol_ether","ketene","carbene","michael_acceptor","alpha_beta_unsaturated",
    "benzene","naphthalene","pyridine","pyrrole","furan","thiophene","imidazole",
    "pyrazole","oxazole","thiazole","pyrimidine","indole","quinoline","isoquinoline",
    "methyl","ethyl","propyl","isopropyl","butyl","isobutyl","sec_butyl","tert_butyl",
    "primary_carbon","secondary_carbon","tertiary_carbon","quaternary_carbon",
    "boc","cbz","fmoc","tosyl","mesyl","triflate","carbonate","carbamate",
    "thioester","imide","sulfonyl_chloride","isonitrile","n_oxide","hydroxylamine",
    "diazo","nitrone","aziridine","silyl_ether","silane","silyl_enol_ether",
    "boronic_acid","boronic_ester","borate","vinyl_ether","orthoester",
    "allyl","benzyl","propargyl"
]

INDEX_MAP["functional_group"] = [f"functional_group_{k}_index" for k in FG_KEYS]
COUNT_MAP["functional_group"] = [f"functional_group_{k}_count" for k in FG_KEYS]
CONSTRAINT_MAP["functional_group"] = [f"functional_group_{k}_nbrInstances" for k in FG_KEYS]

for _fg in FG_KEYS:
    KEY_ALIAS_MAP.setdefault(f"functional_group_{_fg}_count", f"{_fg}_group_count")
    KEY_ALIAS_MAP.setdefault(f"functional_group_{_fg}_index", f"{_fg}_group_indices")
    KEY_ALIAS_MAP.setdefault(f"functional_group_{_fg}_nbrInstances", f"{_fg}_group_instances")

# Reaction success aliasing
for success_key in CONSTRAINT_MAP.get('reaction_success', []):
    base = success_key.replace('template_based_reaction_prediction_', '').replace('_success', '')
    alias = f"{base}_reaction_success"
    KEY_ALIAS_MAP.setdefault(success_key, alias)


# Explicit mapping from count columns to their corresponding index columns
COUNT_TO_INDEX_MAP = {
    "aliphatic_ring_count": "aliphatic_ring_index",
    "aromatic_ring_count": "aromatic_ring_index",
    "branch_point_count": "branch_point_index",
    "brics_decomposition_count": "brics_decomposition_index",
    "bridgehead_count": "bridgehead_index",
    "carbon_atom_count": "carbon_atom_index",
    "chain_termini_count": "chain_termini_index",
    "csp3_carbon_count": "csp3_carbon_index",
    "e_z_stereochemistry_double_bond_e_count": "e_z_stereochemistry_double_bond_e_index",
    "e_z_stereochemistry_double_bond_z_count": "e_z_stereochemistry_double_bond_z_index",
    "fused_ring_count": "fused_ring_index",
    "halogen_atom_count": "halogen_atom_index",
    "hba_count": "hba_index",
    "hbd_count": "hbd_index",
    "heavy_atom_count": "heavy_atom_index",
    "hetero_atom_count": "hetero_atom_index",
    "heterocycle_count": "heterocycle_index",
    "longest_carbon_chain_count": "longest_carbon_chain_index",
    "murcko_scaffold_count": "murcko_scaffold_index",
    "oxidation_state_C_max_count": "oxidation_state_C_max_index",
    "oxidation_state_C_min_count": "oxidation_state_C_min_index",
    "oxidation_state_N_max_count": "oxidation_state_N_max_index",
    "oxidation_state_N_min_count": "oxidation_state_N_min_index",
    "oxidation_state_O_max_count": "oxidation_state_O_max_index",
    "oxidation_state_O_min_count": "oxidation_state_O_min_index",
    "oxidation_state_P_max_count": "oxidation_state_P_max_index",
    "oxidation_state_P_min_count": "oxidation_state_P_min_index",
    "oxidation_state_S_max_count": "oxidation_state_S_max_index",
    "oxidation_state_S_min_count": "oxidation_state_S_min_index",
    "r_s_stereocenter_r_count": "r_s_stereocenter_r_index",
    "r_s_stereocenter_s_count": "r_s_stereocenter_s_index",
    "ring_count": "ring_index",
    "rotatable_bond_count": "rotatable_bond_index",
    "saturated_ring_count": "saturated_ring_index",
    "smallest_largest_ring_size_largest_count": "smallest_largest_ring_size_largest_index",
    "smallest_largest_ring_size_smallest_count": "smallest_largest_ring_size_smallest_index",
    "stereocenter_count": "stereocenter_index",
    "stereochemistry_unspecified_double_bond_count": "stereochemistry_unspecified_double_bond_index",
    "unspecified_stereocenter_count": "unspecified_stereocenter_index",
}

SUBGROUP_DEFINITIONS = {
    "graph_topology": [
        "ring",
        "fused_ring",
        "bridgehead",
        "smallest_largest_ring_size",
        "chain_termini",
        "branch_point",
    ],
    "chemistry_typed_topology": [
        "aromatic_ring",
        "aliphatic_ring",
        "heterocycle",
        "saturated_ring",
        "csp3_carbon",
        "longest_carbon_chain",
        "r_s_stereocenter",
        "unspecified_stereocenter",
        "e_z_stereochemistry_double_bond",
        "stereochemistry_unspecified_double_bond",
        "stereocenter",
    ],
    "composition": [
        "carbon_atom",
        "hetero_atom",
        "halogen_atom",
        "heavy_atom",
        "hydrogen_atom",
        "molecular_formula",
    ],
    "chemical_perception": [
        "hba",
        "hbd",
        "rotatable_bond",
        "oxidation_state",
    ],
    "functional_groups": [
        "functional_group",
        "functional_groups",
    ],
    "synthesis": [
        "brics_decomposition",
        "murcko_scaffold",
        "reaction_success",
    ],
}


def _build_subgroup_map(category_map: Dict[str, list]) -> Dict[str, list]:
    """Construct subgroup mappings ensuring we only return available tasks."""

    subgroup_map: Dict[str, list] = {}
    assigned = set()

    for subgroup, categories in SUBGROUP_DEFINITIONS.items():
        items: list = []
        for category in categories:
            for column in category_map.get(category, []):
                if column not in assigned:
                    items.append(column)
                    assigned.add(column)
        if items:
            subgroup_map[subgroup] = items

    remaining = []
    for columns in category_map.values():
        for column in columns:
            if column not in assigned:
                remaining.append(column)

    if remaining:
        subgroup_map["other"] = remaining

    return subgroup_map


SUBGROUP_COUNT_MAP = _build_subgroup_map(COUNT_MAP)
SUBGROUP_INDEX_MAP = _build_subgroup_map(INDEX_MAP)
SUBGROUP_CONSTRAINT_MAP = _build_subgroup_map(CONSTRAINT_MAP)


# Add functional group count to index mappings
for fg_key in FG_KEYS:
    COUNT_TO_INDEX_MAP[f"functional_group_{fg_key}_count"] = f"functional_group_{fg_key}_index"


ALIAS_TO_KEY_MAP: Dict[str, str] = {}
for key, alias in KEY_ALIAS_MAP.items():
    if isinstance(alias, str):
        ALIAS_TO_KEY_MAP.setdefault(alias, key)
        ALIAS_TO_KEY_MAP.setdefault(alias.lower(), key)

def _register_canonical(key: str):
    ALIAS_TO_KEY_MAP.setdefault(key, key)
    ALIAS_TO_KEY_MAP.setdefault(key.lower(), key)

for mapping in (COUNT_MAP, INDEX_MAP, CONSTRAINT_MAP):
    for props in mapping.values():
        for prop in props:
            _register_canonical(prop)

for alias_map in (COUNT_TO_INDEX_MAP,):
    for src, dst in alias_map.items():
        _register_canonical(src)
        _register_canonical(dst)


def get_alias(property_name: str) -> str:
    """Return the preferred alias for a technical property name."""
    return KEY_ALIAS_MAP.get(property_name, property_name)


def canonicalize_property_name(name: str) -> str:
    """Map an alias (or canonical name) back to the original technical key."""
    if not isinstance(name, str):
        return name
    stripped = name.strip()
    if stripped in ALIAS_TO_KEY_MAP:
        return ALIAS_TO_KEY_MAP[stripped]
    lowered = stripped.lower()
    return ALIAS_TO_KEY_MAP.get(lowered, stripped)
