"""
Mappings based on exact property column names.
Keys use the pattern: property_name + _count or _index
Special cases are handled for compound properties.
Functional groups and reactions are loaded dynamically from files.
"""

from typing import Dict, List, Optional
from pathlib import Path

from ..properties import KEY_ALIAS_MAP
from .._data import SMARTS_RENAMED, REACTION_TEMPLATES as REACTION_TEMPLATES_PATH

# ============================================================================
# DYNAMIC LOADING FUNCTIONS
# ============================================================================

def load_functional_groups(file_path: Optional[Path] = None) -> Dict[str, str]:
    """Load functional groups from smarts_renamed.txt file."""
    groups: Dict[str, str] = {}
    file_path = file_path or SMARTS_RENAMED

    if not file_path.exists():
        raise FileNotFoundError(
            f"Functional group definition file not found at {file_path}."
        )

    with file_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse format: Name:Rank:SMARTS
            parts = line.split(':')
            if len(parts) >= 3:
                name = parts[0]
                # Convert underscore to space for natural language
                natural = name.replace('_', ' ') + " groups"
                groups[name] = natural

    return groups


def load_reaction_templates(file_path: Optional[Path] = None) -> Dict[str, str]:
    """Load reaction templates from reaction_templates.txt file."""
    reactions: Dict[str, str] = {}
    file_path = file_path or REACTION_TEMPLATES_PATH

    if not file_path.exists():
        raise FileNotFoundError(
            f"Reaction template file not found at {file_path}."
        )

    with file_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse format: Name;Category;SMIRKS;Description
            parts = line.split(';')
            if len(parts) >= 4:
                name = parts[0]
                description = parts[3]
                reactions[name] = description

    return reactions


# Load dynamic data
FUNCTIONAL_GROUPS = load_functional_groups()
REACTION_TEMPLATES = load_reaction_templates()

# ============================================================================
# COUNT MAPPINGS - property_name + "_count"
# ============================================================================

COUNT_MAPPINGS = {
    # Ring properties
    "ring_count": ["rings", "number of rings", "ring count", "total rings"],
    "fused_ring_count": ["fused rings", "number of fused rings", "fused ring count", "fused ring systems"],
    "spiro_count": ["spiro centers", "number of spiro centers", "spiro center count", "spiro atoms"],
    "bridgehead_count": ["bridgehead atoms", "number of bridgehead atoms", "bridgehead count", "bridgehead positions"],
    "smallest_ring_size": ["smallest ring size", "size of the smallest ring", "minimum ring size"],
    "largest_ring_size": ["largest ring size", "size of the largest ring", "maximum ring size"],
    "smallest_largest_ring_size_smallest_count": ["atoms making up the smallest ring", "number of atoms in the smallest ring", "size of the smallest ring"],
    "smallest_largest_ring_size_largest_count": ["atoms making up the largest ring", "number of atoms in the largest ring", "size of the largest ring"],
    "aromatic_ring_count": ["aromatic rings", "number of aromatic rings", "aromatic ring count", "aromatic systems"],
    "aliphatic_ring_count": ["aliphatic rings", "number of aliphatic rings", "aliphatic ring count", "non-aromatic rings"],
    "heterocycle_count": ["heterocycles", "number of heterocycles", "heterocyclic ring count", "heterocyclic systems"],
    "saturated_ring_count": ["saturated rings", "number of saturated rings", "saturated ring count", "fully saturated rings"],

    # Chain and structural features
    "chain_termini_count": ["terminal carbons", "chain end carbons", "terminal carbon count"],
    "branch_point_count": ["branch points", "branching point count", "branch carbon count"],
    "longest_carbon_chain": ["length of the longest carbon chain", "maximum carbon chain length", "size of the longest carbon chain"],
    "longest_carbon_chain_count": ["atoms making up the longest carbon chain", "length of the longest carbon chain", "size of the longest carbon chain"],
    "csp3_carbon_count": ["sp3 carbons", "sp³ carbon count", "sp3 hybridized carbon count"],

    # Stereochemistry
    "r_s_stereocenter_r_count": ["R-stereocenters", "R-configured center count", "(R) stereocenter count"],
    "r_s_stereocenter_s_count": ["S-stereocenters", "S-configured center count", "(S) stereocenter count"],
    "unspecified_stereocenter_count": ["unspecified stereocenters", "undefined stereocenter count", "unassigned chiral centers"],
    "e_z_stereochemistry_double_bond_e_count": ["E-double bonds", "E-configured double bond count", "(E) double bonds"],
    "e_z_stereochemistry_double_bond_z_count": ["Z-double bonds", "Z-configured double bond count", "(Z) double bonds"],
    "stereochemistry_unspecified_double_bond_count": ["unspecified E/Z bonds", "undefined double bond stereochemistry", "unassigned E/Z bonds"],
    "stereocenter_count": ["stereocenters", "total chiral center count", "stereogenic center count"],

    # Atoms
    "carbon_atom_count": ["carbon atoms", "number of carbon atoms", "carbon count", "total carbons"],
    "hetero_atom_count": ["heteroatoms", "number of heteroatoms", "heteroatom count", "non-C/H atoms"],
    "halogen_atom_count": ["halogen atoms", "number of halogen atoms", "halogen count", "total halogens"],
    "heavy_atom_count": ["heavy atoms", "number of heavy atoms", "non-hydrogen atom count", "heavy atom count"],
    "hydrogen_atom_count": ["hydrogen atoms", "number of hydrogen atoms", "hydrogen count", "total hydrogens"],

    # Molecular properties
    "molecular_formula": ["molecular formula", "chemical formula", "molecular composition"],
    "molecular_formula_count": ["molecular formula", "chemical formula", "molecular composition"],
    "hba_count": ["hydrogen bond acceptors", "HBA count", "H-bond acceptor sites"],
    "hbd_count": ["hydrogen bond donors", "HBD count", "H-bond donor sites"],
    "rotatable_bond_count": ["rotatable bonds", "rotatable bond count", "freely rotating bonds"],

    # Oxidation states - special handling for each element and state
    "oxidation_state_c_max_count": ["carbons at maximum oxidation", "fully oxidized carbons", "max oxidation C count"],
    "oxidation_state_c_min_count": ["carbons at minimum oxidation", "fully reduced carbons", "min oxidation C count"],
    "oxidation_state_C_max_count": ["carbons at maximum oxidation", "fully oxidized carbons", "max oxidation C count"],
    "oxidation_state_C_min_count": ["carbons at minimum oxidation", "fully reduced carbons", "min oxidation C count"],
    "oxidation_state_n_max_count": ["nitrogens at maximum oxidation", "fully oxidized nitrogens", "max oxidation N count"],
    "oxidation_state_n_min_count": ["nitrogens at minimum oxidation", "fully reduced nitrogens", "min oxidation N count"],
    "oxidation_state_N_max_count": ["nitrogens at maximum oxidation", "fully oxidized nitrogens", "max oxidation N count"],
    "oxidation_state_N_min_count": ["nitrogens at minimum oxidation", "fully reduced nitrogens", "min oxidation N count"],
    "oxidation_state_o_max_count": ["oxygens at maximum oxidation", "fully oxidized oxygens", "max oxidation O count"],
    "oxidation_state_o_min_count": ["oxygens at minimum oxidation", "fully reduced oxygens", "min oxidation O count"],
    "oxidation_state_O_max_count": ["oxygens at maximum oxidation", "fully oxidized oxygens", "max oxidation O count"],
    "oxidation_state_O_min_count": ["oxygens at minimum oxidation", "fully reduced oxygens", "min oxidation O count"],
    "oxidation_state_s_max_count": ["sulfurs at maximum oxidation", "fully oxidized sulfurs", "max oxidation S count"],
    "oxidation_state_s_min_count": ["sulfurs at minimum oxidation", "fully reduced sulfurs", "min oxidation S count"],
    "oxidation_state_S_max_count": ["sulfurs at maximum oxidation", "fully oxidized sulfurs", "max oxidation S count"],
    "oxidation_state_S_min_count": ["sulfurs at minimum oxidation", "fully reduced sulfurs", "min oxidation S count"],
    "oxidation_state_p_max_count": ["phosphorus atoms at maximum oxidation", "fully oxidized phosphorus atoms", "max oxidation P count"],
    "oxidation_state_p_min_count": ["phosphorus atoms at minimum oxidation", "fully reduced phosphorus atoms", "min oxidation P count"],
    "oxidation_state_P_max_count": ["phosphorus atoms at maximum oxidation", "fully oxidized phosphorus atoms", "max oxidation P count"],
    "oxidation_state_P_min_count": ["phosphorus atoms at minimum oxidation", "fully reduced phosphorus atoms", "min oxidation P count"],


    # BRICS and scaffold
    "brics_decomposition_count": ["BRICS fragments", "BRICS fragment count", "BRICS decomposition pieces"],
    "murcko_scaffold_count": ["atoms in Murcko scaffold", "Murcko scaffold size", "scaffold atom count"],

    # Template reactions (count of applicable templates)
    "template_based_reaction_prediction_count": ["number of applicable reaction templates", "reaction template count", "possible reaction sites"],
}

# Dynamically add functional group atom counts (total atoms in all instances)
for fg_name, fg_natural in FUNCTIONAL_GROUPS.items():
    key = f"functional_group_{fg_name}_count"
    fg_display = fg_name.replace('_', ' ')
    natural_forms = [
        f"atoms in {fg_display} groups",
        f"{fg_display} group atoms",
        f"total {fg_display} atoms",
        f"{fg_display} atom count"
    ]
    COUNT_MAPPINGS[key] = natural_forms

# Add functional group instance counts (number of occurrences) for constraints
for fg_name in FUNCTIONAL_GROUPS.keys():
    key = f"functional_group_{fg_name}_nbrInstances"
    fg_display = fg_name.replace('_', ' ')
    natural_forms = [
        f"{fg_display} groups",
        f"number of {fg_display} groups",
        f"{fg_display} group count",
        f"{fg_display} instances"
    ]
    COUNT_MAPPINGS[key] = natural_forms

# Dynamically add reaction template counts
for reaction_name, reaction_desc in REACTION_TEMPLATES.items():
    key = f"reaction_{reaction_name}_count"
    # Create cleaner natural language
    clean_name = reaction_name.replace('_', ' ')
    COUNT_MAPPINGS[key] = [
        f"{clean_name} sites",  # Simplest form first
        f"number of {clean_name} sites",
        f"{clean_name} reaction count"
    ]

    success_key = f"template_based_reaction_prediction_{reaction_name}_success"
    success_desc = reaction_desc or clean_name
    if success_desc and success_desc[0].isupper():
        success_desc = success_desc[0].lower() + success_desc[1:]
    COUNT_MAPPINGS.setdefault(success_key, [
        f"success of {success_desc}",
        f"{success_desc} success",
        f"{success_desc} outcome"
    ])

# Add template-based reaction prediction mappings for moleculariq properties
TEMPLATE_REACTION_MAPPINGS = {
    "acetylation": "acetylation reaction",
    "acid_chloride_to_amide": "acid chloride to amide conversion",
    "acid_chloride_to_ester_methyl": "acid chloride to methyl ester conversion",
    "acid_to_acid_chloride": "acid to acid chloride conversion",
    "alcohol_to_alkyl_halide_Br": "alcohol to alkyl bromide conversion",
    "alcohol_to_alkyl_halide_Cl": "alcohol to alkyl chloride conversion",
    "alcohol_to_alkyl_halide_I": "alcohol to alkyl iodide conversion",
    "alcohol_to_mesylate": "alcohol to mesylate conversion",
    "alcohol_to_tosylate": "alcohol to tosylate conversion",
    "aldehyde_to_carboxylic_acid": "aldehyde oxidation to carboxylic acid",
    "aldehyde_to_primary_alcohol": "aldehyde reduction to primary alcohol",
    "alkene_to_alkane": "alkene hydrogenation to alkane",
    "alkyl_halide_to_alcohol": "alkyl halide substitution to alcohol",
    "alkyl_halide_to_azide": "alkyl halide to azide substitution",
    "alkyl_halide_to_nitrile": "alkyl halide to nitrile substitution",
    "alkyne_to_alkene": "alkyne reduction to alkene",
    "amide_hydrolysis_to_acid": "amide hydrolysis to carboxylic acid",
    "amide_to_amine": "amide reduction to amine",
    "aromatic_amine_to_diazonium": "aromatic amine diazotization",
    "aromatic_fluorination": "aromatic fluorination reaction",
    "aromatic_iodination": "aromatic iodination reaction",
    "aromatic_sulfonation": "aromatic sulfonation reaction",
    "azide_to_amine": "azide reduction to amine",
    "beckmann_rearrangement": "Beckmann rearrangement reaction",
    "benzylic_alcohol_oxidation": "benzylic alcohol oxidation",
    "benzylic_oxidation": "benzylic position oxidation",
    "boc_deprotection": "Boc protecting group removal",
    "boc_protection": "Boc protecting group addition",
    "bromination": "bromination reaction",
    "carboxylic_acid_to_primary_alcohol": "carboxylic acid reduction to primary alcohol",
    "chlorination": "chlorination reaction",
    "deacetylation": "acetyl group removal",
    "decarboxylation": "carboxyl group removal",
    "dehydration_alcohol": "alcohol dehydration to alkene",
    "dehydrohalogenation": "dehydrohalogenation elimination",
    "dibromination": "dibromination addition",
    "dichlorination": "dichlorination addition",
    "epoxidation": "alkene epoxidation",
    "epoxide_to_diol": "epoxide ring opening to diol",
    "epoxide_to_halohydrin_Br": "epoxide opening to bromohydrin",
    "epoxide_to_halohydrin_Cl": "epoxide opening to chlorohydrin",
    "ester_hydrolysis_to_acid": "ester hydrolysis to carboxylic acid",
    "ester_to_alcohol": "ester reduction to alcohol",
    "hydration_of_alkene": "alkene hydration to alcohol",
    "hydrazone_formation": "hydrazone formation from carbonyl",
    "hydrohalogenation_HBr": "hydrobromic acid addition to alkene",
    "hydrohalogenation_HCl": "hydrochloric acid addition to alkene",
    "imine_to_amine": "imine reduction to amine",
    "ketone_to_secondary_alcohol": "ketone reduction to secondary alcohol",
    "nitration": "nitration reaction",
    "nitrile_hydrolysis": "nitrile hydrolysis to carboxylic acid",
    "nitrile_to_amine": "nitrile reduction to amine",
    "nitro_to_amine": "nitro group reduction to amine",
    "oxime_formation": "oxime formation from carbonyl",
    "ozonolysis_terminal": "terminal alkene ozonolysis",
    "primary_alcohol_to_aldehyde": "primary alcohol oxidation to aldehyde",
    "secondary_alcohol_to_ketone": "secondary alcohol oxidation to ketone",
    "sulfide_to_sulfoxide": "sulfide oxidation to sulfoxide",
}

# Add template reaction prediction counts
for reaction_key, reaction_desc in TEMPLATE_REACTION_MAPPINGS.items():
    count_key = f"template_based_reaction_prediction_{reaction_key}_count"
    COUNT_MAPPINGS[count_key] = [
        f"sites for {reaction_desc}",
        f"number of {reaction_desc} sites",
        f"{reaction_desc} sites",
        f"positions for {reaction_desc}"
    ]

# ============================================================================
# INDEX MAPPINGS - property_name + "_index"
# ============================================================================

INDEX_MAPPINGS = {
    # Ring atoms
    "ring_index": ["atoms in rings", "ring atoms", "ring atom indices"],
    "fused_ring_index": ["atoms in fused rings", "fused ring atoms", "fused system atoms"],
    "spiro_index": ["spiro centers", "spiro atom indices", "spiro locations"],
    "bridgehead_index": ["bridgehead atoms", "bridgehead indices", "bridgehead locations"],
    "smallest_ring_index": ["atoms in the smallest ring", "smallest ring atoms", "minimum ring atoms"],
    "largest_ring_index": ["atoms in the largest ring", "largest ring atoms", "maximum ring atoms"],
    "smallest_largest_ring_size_smallest_index": ["atoms in the smallest ring", "smallest ring atoms", "smallest ring indices"],
    "smallest_largest_ring_size_largest_index": ["atoms in the largest ring", "largest ring atoms", "largest ring indices"],
    "aromatic_ring_index": ["atoms in aromatic rings", "aromatic ring atoms", "aromatic system atoms"],
    "aliphatic_ring_index": ["atoms in aliphatic rings", "aliphatic ring atoms", "non-aromatic ring atoms"],
    "heterocycle_index": ["atoms in heterocycles", "heterocycle atoms", "heterocyclic atoms"],
    "saturated_ring_index": ["atoms in saturated rings", "saturated ring atoms", "fully saturated ring atoms"],

    # Chain and structural positions
    "chain_termini_index": ["terminal carbons", "chain end indices", "terminal atoms"],
    "branch_point_index": ["branch points", "branching carbon indices", "branch atoms"],
    "longest_carbon_chain_index": ["atoms in the longest carbon chain", "longest chain atoms", "longest C-chain atoms"],
    "csp3_carbon_index": ["sp3 carbons", "sp³ carbon indices", "sp3 hybridized atoms"],

    # Stereochemistry positions
    "r_s_stereocenter_r_index": ["R-stereocenters", "R-configured atom indices", "(R) chiral centers"],
    "r_s_stereocenter_s_index": ["S-stereocenters", "S-configured atom indices", "(S) chiral centers"],
    "unspecified_stereocenter_index": ["unspecified stereocenters", "undefined chiral center indices", "unassigned stereocenter atoms"],
    "e_z_stereochemistry_double_bond_e_index": ["atoms in E-double bonds", "E-configured bond atoms", "(E) double bond atoms"],
    "e_z_stereochemistry_double_bond_z_index": ["atoms in Z-double bonds", "Z-configured bond atoms", "(Z) double bond atoms"],
    "stereochemistry_unspecified_double_bond_index": ["atoms in unspecified E/Z bonds", "undefined double bond atoms", "unassigned E/Z atoms"],
    "stereocenter_index": ["stereocenters", "chiral center indices", "stereogenic atoms"],

    # Atom positions
    "carbon_atom_index": ["carbon atoms", "carbon indices", "C atom locations"],
    "hetero_atom_index": ["heteroatoms", "heteroatom indices", "non-C/H atoms"],
    "halogen_atom_index": ["halogen atoms", "halogen indices", "halogen locations"],
    "heavy_atom_index": ["heavy atoms", "non-hydrogen indices", "heavy atom locations"],
    "hydrogen_atom_index": ["hydrogen atoms", "hydrogen indices", "H atom locations"],

    # Property positions
    "hba_index": ["hydrogen bond acceptors", "HBA sites", "H-bond acceptor atoms"],
    "hbd_index": ["hydrogen bond donors", "HBD sites", "H-bond donor atoms"],
    "rotatable_bond_index": ["rotatable bonds", "rotatable bond atoms", "freely rotating bond atoms"],


    # Oxidation state positions
    "oxidation_state_c_max_index": ["carbons at maximum oxidation", "fully oxidized carbons", "max oxidation C atoms"],
    "oxidation_state_c_min_index": ["carbons at minimum oxidation", "fully reduced carbons", "min oxidation C atoms"],
    "oxidation_state_C_max_index": ["carbons at maximum oxidation", "fully oxidized carbons", "max oxidation C atoms"],
    "oxidation_state_C_min_index": ["carbons at minimum oxidation", "fully reduced carbons", "min oxidation C atoms"],
    "oxidation_state_n_max_index": ["nitrogens at maximum oxidation", "fully oxidized nitrogens", "max oxidation N atoms"],
    "oxidation_state_n_min_index": ["nitrogens at minimum oxidation", "fully reduced nitrogens", "min oxidation N atoms"],
    "oxidation_state_N_max_index": ["nitrogens at maximum oxidation", "fully oxidized nitrogens", "max oxidation N atoms"],
    "oxidation_state_N_min_index": ["nitrogens at minimum oxidation", "fully reduced nitrogens", "min oxidation N atoms"],
    "oxidation_state_o_max_index": ["oxygens at maximum oxidation", "fully oxidized oxygens", "max oxidation O atoms"],
    "oxidation_state_o_min_index": ["oxygens at minimum oxidation", "fully reduced oxygens", "min oxidation O atoms"],
    "oxidation_state_O_max_index": ["oxygens at maximum oxidation", "fully oxidized oxygens", "max oxidation O atoms"],
    "oxidation_state_O_min_index": ["oxygens at minimum oxidation", "fully reduced oxygens", "min oxidation O atoms"],
    "oxidation_state_s_max_index": ["sulfurs at maximum oxidation", "fully oxidized sulfurs", "max oxidation S atoms"],
    "oxidation_state_s_min_index": ["sulfurs at minimum oxidation", "fully reduced sulfurs", "min oxidation S atoms"],
    "oxidation_state_S_max_index": ["sulfurs at maximum oxidation", "fully oxidized sulfurs", "max oxidation S atoms"],
    "oxidation_state_S_min_index": ["sulfurs at minimum oxidation", "fully reduced sulfurs", "min oxidation S atoms"],
    "oxidation_state_p_max_index": ["phosphorus at maximum oxidation", "fully oxidized phosphorus atoms", "max oxidation P atoms"],
    "oxidation_state_p_min_index": ["phosphorus at minimum oxidation", "fully reduced phosphorus atoms", "min oxidation P atoms"],
    "oxidation_state_P_max_index": ["phosphorus at maximum oxidation", "fully oxidized phosphorus atoms", "max oxidation P atoms"],
    "oxidation_state_P_min_index": ["phosphorus at minimum oxidation", "fully reduced phosphorus atoms", "min oxidation P atoms"],


    # BRICS and scaffold positions
    "brics_decomposition_index": ["atoms at BRICS breakpoints", "BRICS bond atoms", "BRICS cleavage sites"],
    "murcko_scaffold_index": ["atoms in Murcko scaffold", "scaffold atoms", "core structure atoms"],

    # Template reaction sites
    "template_based_reaction_prediction_index": ["reactive sites", "reaction template atoms", "potential reaction centers"],
}

# Dynamically add functional group indices (atom positions)
for fg_name, fg_natural in FUNCTIONAL_GROUPS.items():
    key = f"functional_group_{fg_name}_index"
    fg_display = fg_name.replace('_', ' ')
    natural_forms = [
        f"{fg_display} atom positions",
        f"{fg_display} atom indices",
        f"indices of {fg_display} atoms",
        f"positions of atoms in {fg_display} groups"
    ]
    INDEX_MAPPINGS[key] = natural_forms

# Dynamically add reaction template indices
for reaction_name, reaction_desc in REACTION_TEMPLATES.items():
    key = f"reaction_{reaction_name}_index"
    clean_name = reaction_name.replace('_', ' ')
    INDEX_MAPPINGS[key] = [
        f"atoms at {clean_name} sites",
        f"{clean_name} reaction atoms",
        f"{clean_name} reactive positions"
    ]

# Add template reaction prediction indices for moleculariq properties
for reaction_key, reaction_desc in TEMPLATE_REACTION_MAPPINGS.items():
    index_key = f"template_based_reaction_prediction_{reaction_key}_index"
    INDEX_MAPPINGS[index_key] = [
        f"atoms involved in {reaction_desc}",
        f"reactive atoms for {reaction_desc}",
        f"{reaction_desc} atom positions",
        f"positions for {reaction_desc}"
    ]

# ============================================================================
# ALIASES - Common variations that map to the standard keys
# ============================================================================

_BASE_ALIASES = {
    # Simple ring variations
    "rings": "ring_count",
    "ring count": "ring_count",
    "number of rings": "ring_count",

    # Aromatic variations
    "aromatic rings": "aromatic_ring_count",
    "aromatic ring count": "aromatic_ring_count",
    "aromatic ring atoms": "aromatic_ring_index",
    "aromatic atoms": "aromatic_ring_index",

    # Stereocenter variations
    "stereocenters": "stereocenter_count",
    "chiral centers": "stereocenter_count",
    "chiral atoms": "stereocenter_index",
    "r stereocenters": "r_s_stereocenter_r_count",
    "s stereocenters": "r_s_stereocenter_s_count",
    "r-stereocenters": "r_s_stereocenter_r_count",
    "s-stereocenters": "r_s_stereocenter_s_count",
    "(r) stereocenters": "r_s_stereocenter_r_count",
    "(s) stereocenters": "r_s_stereocenter_s_count",

    # Double bond stereochemistry
    "e double bonds": "e_z_stereochemistry_double_bond_e_count",
    "z double bonds": "e_z_stereochemistry_double_bond_z_count",
    "e-double bonds": "e_z_stereochemistry_double_bond_e_count",
    "z-double bonds": "e_z_stereochemistry_double_bond_z_count",
    "(e) double bonds": "e_z_stereochemistry_double_bond_e_count",
    "(z) double bonds": "e_z_stereochemistry_double_bond_z_count",

    # Atom variations
    "carbon": "carbon_atom_count",
    "carbon atoms": "carbon_atom_count",
    "carbon positions": "carbon_atom_index",
    "carbons": "carbon_atom_count",
    "c atoms": "carbon_atom_count",
    "nitrogen": "nitrogen_atom_count",
    "nitrogens": "nitrogen_atom_count",
    "oxygen": "oxygen_atom_count",
    "oxygens": "oxygen_atom_count",
    "sulfur": "sulfur_atom_count",
    "sulfurs": "sulfur_atom_count",
    "heteroatoms": "hetero_atom_count",
    "heteroatom positions": "hetero_atom_index",
    "halogens": "halogen_atom_count",
    "halogen positions": "halogen_atom_index",
    "heavy atoms": "heavy_atom_count",
    "non-hydrogen atoms": "heavy_atom_count",
    "hydrogens": "hydrogen_atom_count",
    "h atoms": "hydrogen_atom_count",

    # HBA/HBD variations
    "hydrogen bond acceptors": "hba_count",
    "hydrogen bond donors": "hbd_count",
    "h-bond acceptors": "hba_count",
    "h-bond donors": "hbd_count",
    "hba": "hba_count",
    "hbd": "hbd_count",
    "hba sites": "hba_index",
    "hbd sites": "hbd_index",

    # Rotatable bonds
    "rotatable bonds": "rotatable_bond_count",
    "freely rotating bonds": "rotatable_bond_count",
    "rotatable bond atoms": "rotatable_bond_index",

    # Chain features
    "terminal carbons": "chain_termini_count",
    "terminal carbon atoms": "chain_termini_index",
    "chain ends": "chain_termini_count",
    "branch points": "branch_point_count",
    "branching points": "branch_point_count",
    "branching carbons": "branch_point_index",

    # Hybridization
    "sp3 carbons": "csp3_carbon_count",
    "sp³ carbons": "csp3_carbon_count",
    "sp3 carbon atoms": "csp3_carbon_index",

    # Ring size
    "smallest ring": "smallest_ring_size",
    "largest ring": "largest_ring_size",
    "minimum ring size": "smallest_ring_size",
    "maximum ring size": "largest_ring_size",

    # Functional groups - common short forms
    "alcohols": "functional_group_alcohol_count",
    "ketones": "functional_group_ketone_count",
    "aldehydes": "functional_group_aldehyde_count",
    "carboxylic acids": "functional_group_carboxylic_acid_count",
    "esters": "functional_group_ester_count",
    "amides": "functional_group_amide_count",
    "amines": "functional_group_amine_count",
    "primary amines": "functional_group_primary_amine_count",
    "secondary amines": "functional_group_secondary_amine_count",
    "tertiary amines": "functional_group_tertiary_amine_count",
    "phenols": "functional_group_phenol_count",
    "thiols": "functional_group_thiol_count",
    "ethers": "functional_group_ether_count",
    "epoxides": "functional_group_epoxide_count",
    "alkenes": "functional_group_alkene_count",
    "alkynes": "functional_group_alkyne_count",
    "nitro groups": "functional_group_nitro_count",
    "nitrile groups": "functional_group_nitrile_count",
    "cyano groups": "functional_group_nitrile_count",

    # BRICS
    "brics fragments": "brics_decomposition_count",
    "brics bonds": "brics_decomposition_index",
    "brics breakpoints": "brics_decomposition_index",

    # Murcko
    "murcko scaffold": "murcko_scaffold_count",
    "scaffold size": "murcko_scaffold_count",
    "scaffold atoms": "murcko_scaffold_index",

    # Oxidation states - common forms
    "oxidized carbons": "oxidation_state_c_max_count",
    "reduced carbons": "oxidation_state_c_min_count",
    "oxidized carbon positions": "oxidation_state_c_max_index",
    "reduced carbon positions": "oxidation_state_c_min_index",
}

ALIASES: Dict[str, str] = {key.lower(): value for key, value in _BASE_ALIASES.items()}

# Dynamically add aliases for functional groups
for fg_name in FUNCTIONAL_GROUPS.keys():
    # Add plurals as aliases for count
    if fg_name.endswith('e'):
        plural = fg_name[:-1] + 's'  # e.g., amine -> amines
    elif fg_name.endswith('y'):
        plural = fg_name[:-1] + 'ies'  # e.g., epoxy -> epoxies
    else:
        plural = fg_name + 's'  # e.g., alcohol -> alcohols

    ALIASES[plural.lower()] = f"functional_group_{fg_name}_count"

    # Also add common variations
    name_with_spaces = fg_name.replace('_', ' ')
    ALIASES[name_with_spaces.lower()] = f"functional_group_{fg_name}_count"
    ALIASES[f"{name_with_spaces} groups".lower()] = f"functional_group_{fg_name}_count"
    ALIASES[f"{name_with_spaces} atoms".lower()] = f"functional_group_{fg_name}_index"

# Add comprehensive aliases for better coverage
_COMPREHENSIVE_ALIASES = {
    # Molecular formula (identity mapping)
    "molecular_formula": "molecular_formula",
    "molecular formula": "molecular_formula",

    # Benzene/phenyl/aromatic variations
    "benzene rings": "aromatic_ring_count",
    "benzene ring": "aromatic_ring_count",
    "phenyl rings": "aromatic_ring_count",
    "phenyl ring": "aromatic_ring_count",
    "aromatic ring systems": "aromatic_ring_count",
    "cyclic structures": "ring_count",

    # British spellings and stereochemistry
    "chiral centres": "stereocenter_count",
    "stereogenic centres": "stereocenter_count",
    "asymmetric centers": "stereocenter_count",
    "asymmetric centres": "stereocenter_count",
    "optical centers": "stereocenter_count",
    "optical centres": "stereocenter_count",

    # Configuration variations
    "r configuration": "r_s_stereocenter_r_count",
    "s configuration": "r_s_stereocenter_s_count",
    "cis double bonds": "e_z_stereochemistry_double_bond_z_count",
    "trans double bonds": "e_z_stereochemistry_double_bond_e_count",
    "e configuration": "e_z_stereochemistry_double_bond_e_count",
    "z configuration": "e_z_stereochemistry_double_bond_z_count",

    # Chemical group shortcuts (both cases)
    "hydroxyl groups": "functional_group_alcohol_count",
    "hydroxyl": "functional_group_alcohol_count",
    "oh groups": "functional_group_alcohol_count",
    "OH groups": "functional_group_alcohol_count",
    "oh": "functional_group_alcohol_count",
    "OH": "functional_group_alcohol_count",
    "carbonyl": "functional_group_ketone_count",
    "carbonyl groups": "functional_group_ketone_count",
    "c=o": "functional_group_ketone_count",
    "C=O": "functional_group_ketone_count",
    "c=o groups": "functional_group_ketone_count",
    "C=O groups": "functional_group_ketone_count",
    "amino": "functional_group_amine_count",
    "amino groups": "functional_group_amine_count",
    "nh2": "functional_group_primary_amine_count",
    "NH2": "functional_group_primary_amine_count",
    "nh2 groups": "functional_group_primary_amine_count",
    "NH2 groups": "functional_group_primary_amine_count",
    "nh": "functional_group_secondary_amine_count",
    "NH": "functional_group_secondary_amine_count",
    "nh groups": "functional_group_secondary_amine_count",
    "NH groups": "functional_group_secondary_amine_count",
    "cooh": "functional_group_carboxylic_acid_count",
    "COOH": "functional_group_carboxylic_acid_count",
    "cooh groups": "functional_group_carboxylic_acid_count",
    "COOH groups": "functional_group_carboxylic_acid_count",
    "carboxyl": "functional_group_carboxylic_acid_count",
    "carboxyl groups": "functional_group_carboxylic_acid_count",
    "sh": "functional_group_thiol_count",
    "SH": "functional_group_thiol_count",
    "sh groups": "functional_group_thiol_count",
    "SH groups": "functional_group_thiol_count",
    "mercapto": "functional_group_thiol_count",
    "mercapto groups": "functional_group_thiol_count",

    # Element shortcuts (both cases and variations)
    "c": "carbon_atom_count",
    "C": "carbon_atom_count",
    "C atoms": "carbon_atom_count",
    "n": "nitrogen_atom_count",
    "N": "nitrogen_atom_count",
    "n atoms": "nitrogen_atom_count",
    "N atoms": "nitrogen_atom_count",
    "o": "oxygen_atom_count",
    "O": "oxygen_atom_count",
    "o atoms": "oxygen_atom_count",
    "O atoms": "oxygen_atom_count",
    "s": "sulfur_atom_count",
    "S": "sulfur_atom_count",
    "s atoms": "sulfur_atom_count",
    "S atoms": "sulfur_atom_count",
    "p": "phosphorus_atom_count",
    "P": "phosphorus_atom_count",
    "p atoms": "phosphorus_atom_count",
    "P atoms": "phosphorus_atom_count",
    "f": "fluorine_atom_count",
    "F": "fluorine_atom_count",
    "f atoms": "fluorine_atom_count",
    "cl": "chlorine_atom_count",
    "Cl": "chlorine_atom_count",
    "cl atoms": "chlorine_atom_count",
    "br": "bromine_atom_count",
    "Br": "bromine_atom_count",
    "br atoms": "bromine_atom_count",
    "i": "iodine_atom_count",
    "I": "iodine_atom_count",
    "i atoms": "iodine_atom_count",
    "fluorine": "fluorine_atom_count",
    "chlorine": "chlorine_atom_count",
    "bromine": "bromine_atom_count",
    "iodine": "iodine_atom_count",
    "non-carbon atoms": "hetero_atom_count",

    # Property variations
    "h-bond acceptors": "hba_count",
    "h bond acceptors": "hba_count",
    "hbas": "hba_count",
    "HBAs": "hba_count",
    "h-bond donors": "hbd_count",
    "h bond donors": "hbd_count",
    "hbds": "hbd_count",
    "HBDs": "hbd_count",
    "freely rotatable bonds": "rotatable_bond_count",
    "single bonds that can rotate": "rotatable_bond_count",

    # Ring/cycle variations
    "ring atoms": "ring_index",
    "cyclic atoms": "ring_index",
    "atoms in rings": "ring_index",
    "number of atoms in rings": "ring_index",
    "cyclic": "ring_count",
    "ring size": "ring_count",
    "ring sizes": "ring_count",

    # BRICS and Murcko variations
    "BRICS fragments": "brics_decomposition_count",
    "retrosynthetic fragments": "brics_decomposition_count",
    "scaffold": "murcko_scaffold_count",
    "core structure": "murcko_scaffold_count",
    "molecular scaffold": "murcko_scaffold_count",
}

# Merge comprehensive aliases
ALIASES.update({key.lower(): value for key, value in _COMPREHENSIVE_ALIASES.items()})

# Register technical aliases defined in the benchmark column map
for technical_key, alias in KEY_ALIAS_MAP.items():
    if not isinstance(alias, str):
        continue
    alias_lower = alias.lower()
    ALIASES.setdefault(alias_lower, technical_key)

    if technical_key in COUNT_MAPPINGS:
        options = COUNT_MAPPINGS[technical_key]
        if isinstance(options, list) and alias not in options:
            options.append(alias)
    if technical_key in INDEX_MAPPINGS:
        options = INDEX_MAPPINGS[technical_key]
        if isinstance(options, list) and alias not in options:
            options.append(alias)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _build_reverse_lookup() -> Dict[str, str]:
    """Create a lookup from natural language strings to technical keys."""
    reverse: Dict[str, str] = {}

    for technical_key, natural_forms in COUNT_MAPPINGS.items():
        reverse.setdefault(technical_key.lower(), technical_key)
        reverse.setdefault(technical_key.replace('_', ' '), technical_key)
        for natural_form in natural_forms:
            normalized = natural_form.lower().strip()
            if normalized and normalized not in reverse:
                reverse[normalized] = technical_key

    for technical_key, natural_forms in INDEX_MAPPINGS.items():
        reverse.setdefault(technical_key.lower(), technical_key)
        reverse.setdefault(technical_key.replace('_', ' '), technical_key)
        for natural_form in natural_forms:
            normalized = natural_form.lower().strip()
            if normalized and normalized not in reverse:
                reverse[normalized] = technical_key

    # Aliases take precedence and may intentionally override defaults
    reverse.update(ALIASES)

    return reverse


_REVERSE_LOOKUP = _build_reverse_lookup()

def get_natural_language(technical_key: str, context: str = "count") -> List[str]:
    """
    Get natural language forms for a technical key.

    Args:
        technical_key: The technical property name
        context: "count", "index", or "constraint"

    Returns:
        List of natural language variations
    """
    if context == "count":
        return COUNT_MAPPINGS.get(technical_key, [technical_key])
    elif context == "index":
        return INDEX_MAPPINGS.get(technical_key, [technical_key])
    elif context == "constraint":
        # For constraints, use COUNT_MAPPINGS as default since most constraints are counts
        return COUNT_MAPPINGS.get(technical_key, [technical_key])
    else:
        return [technical_key]


def parse_natural_language(natural_text: str) -> str:
    """
    Parse natural language to technical key.

    Args:
        natural_text: Natural language description

    Returns:
        Technical key or original text if not found
    """
    normalized = " ".join(natural_text.lower().split())
    if not normalized:
        return natural_text

    return _REVERSE_LOOKUP.get(normalized, natural_text)


def get_all_properties() -> Dict[str, List[str]]:
    """
    Get all properties organized by type.

    Returns:
        Dictionary with "count" and "index" keys containing property lists
    """
    return {
        "count": list(COUNT_MAPPINGS.keys()),
        "index": list(INDEX_MAPPINGS.keys())
    }
