"""
Solver for template-based reaction tasks:
- Reactions are provided via reaction templates
- The solver evaluates:
    a) whether a given reaction template can be applied to a given molecule
    b) which products are obtained by applying the reaction template
"""

#---------------------------------------------------------------------------------------
# Config
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from .._data import REACTION_TEMPLATES

@dataclass
class TemplateBasedReactionSolverConfig:
    reaction_templates_path: Path = REACTION_TEMPLATES

#---------------------------------------------------------------------------------------
# Imports
from typing import Optional, Dict
from rdkit import Chem
from rdkit.Chem import AllChem
#---------------------------------------------------------------------------------------
# Class definitions

class TemplateBasedReactionSolver:
    def __init__(self, 
                 config: TemplateBasedReactionSolverConfig=TemplateBasedReactionSolverConfig()):
        self.config = config
        
        # Initialize and load reaction templates
        self.templates = self._load_reaction_templates(config.reaction_templates_path)
        
        # Pre-compile reaction templates
        self.compiled_templates = self._compile_reaction_templates()
        
        # Pre-compile reaction center patterns
        self.reaction_center_patterns = self._get_reaction_center_patterns()
        self.compiled_center_patterns = self._compile_center_patterns()
    
    def _apply_reaction(self, smiles: str, reaction) -> Optional[str]:
        """Apply a single reaction to a molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Apply reaction
        products = reaction.RunReactants((mol,))

        if products and len(products) > 0:
            product_list = list()
            for i in range(len(products)):
                product_mol = products[i][0]
                # Convert to SMILES
                product_smiles = Chem.MolToSmiles(product_mol)
                product_list.append(product_smiles)
            return product_list
        return None
    
    def _get_center_counts_and_indices(self, smiles: str) -> Dict[str, Dict[str, any]]:
        """Get atom indices involved in the reaction center."""

        results = {}

        # Parse molecule once
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            for reaction_name, _ in self.compiled_center_patterns.items():
                results[f"template_based_reaction_prediction_{reaction_name}_count"] = 0
                results[f"template_based_reaction_prediction_{reaction_name}_index"] = []
            return results

        # Process all patterns
        for reaction_name, pattern in self.compiled_center_patterns.items():
            # Collect all atom indices and count instances for this functional group
            atom_indices = set()

            matches = mol.GetSubstructMatches(pattern)
            count = len(matches)

            if count > 0:
                for match in matches:
                    atom_indices.update(match)
                indices_list = sorted(list(atom_indices))
            else:
                indices_list = []

            results[f"template_based_reaction_prediction_{reaction_name}_count"] = count
            results[f"template_based_reaction_prediction_{reaction_name}_index"] = indices_list

        return results
        
    def get_reaction_data(self, smiles: str) -> dict:
        """
        Evaluates all pre-compiled reaction templates on the given molecule.
        """
        
        results = self._get_center_counts_and_indices(smiles)
        
        for name, reaction in self.compiled_templates.items():
            product_list = self._apply_reaction(smiles, reaction)
            assert product_list is None or isinstance(product_list, list)
            success = 0 if product_list is None else 1
            results[f"template_based_reaction_prediction_{name}_success"] = success
            results[f"template_based_reaction_prediction_{name}_products"] = product_list if success else None
        return results
    
    #-----------------------------------------------------------------------------------
    # General utility methods
    def _load_reaction_templates(self, reaction_file: str):
        """Load reaction templates from file."""
        templates = {}
        with open(reaction_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(';')
                    if len(parts) >= 4:
                        name = parts[0]
                        smirks = parts[2]
                        templates[name] = smirks
        return templates
    
    def _compile_reaction_templates(self):
        """Pre-compile reaction templates into RDKit reaction objects."""
        compiled = {}
        for name, smirks in self.templates.items():
            try:
                rxn = AllChem.ReactionFromSmarts(smirks)
                if rxn:
                    compiled[name] = rxn
            except Exception as e:
                raise ValueError(f"Failed to compile reaction {name}: {e}")
        return compiled
    
    def _get_reaction_center_patterns(self):
        """
        Extract reaction center SMARTS patterns from reaction templates.
        """
        
        patterns = {}
        for name, smirks in self.templates.items():
            try:
                # Split into reactant and product SMARTS
                reactant_smarts, product_smarts = smirks.split('>>')
                patterns[name] = reactant_smarts
            except Exception as e:
                raise ValueError(f"Failed to extract pattern from {name}: {e}")
        return patterns
    
    def _compile_center_patterns(self):
        """Pre-compile reaction center SMARTS patterns."""
        compiled = {}
        for name, smarts in self.reaction_center_patterns.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    compiled[name] = pattern
            except Exception as e:
                raise ValueError(f"Failed to compile pattern for {name}: {e}")
        return compiled
    
#---------------------------------------------------------------------------------------
# Debugging

if __name__ == "__main__":
    solver = TemplateBasedReactionSolver()
    
    smiles = "OCCCCO"
    results = solver.get_reaction_data(smiles)
    print(results)

    
    