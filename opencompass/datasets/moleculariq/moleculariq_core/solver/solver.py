"""
This file includes a symbolic solver class which is able to compute all ground truth
values for count and indices tasks.
"""

#---------------------------------------------------------------------------------------
# Imports
from .functional_group_solver import FunctionalGroupSolver
from .template_based_reaction_solver import TemplateBasedReactionSolver

from typing import List, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem import Descriptors
from rdkit.Chem import BRICS
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem
from collections import deque
#---------------------------------------------------------------------------------------
# Class definitions

STRICT_ROTATABLE_BOND_SMARTS = (
    "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)"
    "&!$(C([CH3])([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])"
    "&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])"
    "&!$([#7!D1]-!@[CD3]=[N+])]-,:;!@[!$(*#*)&!D1&!$(C(F)(F)F)"
    "&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]"
)
STRICT_ROTATABLE_BOND_PATTERN = Chem.MolFromSmarts(STRICT_ROTATABLE_BOND_SMARTS)

class SymbolicSolver:
    def __init__(self):
        self.functional_group_solver = FunctionalGroupSolver()
        self.reaction_solver = TemplateBasedReactionSolver()

    #-----------------------------------------------------------------------------------
    # Graph topology
    
    # - Count tasks
    def get_ring_count(self, smiles: str) -> int:
        """Get total number of rings."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        ring_info = mol.GetRingInfo()
        return ring_info.NumRings()

    def get_fused_ring_count(self, smiles: str) -> int:
        """Get count of fused ring systems."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        
        if len(atom_rings) < 2:
            return 0
        
        # Find fused systems (rings that share 2+ atoms)
        fused_systems = []
        for i in range(len(atom_rings)):
            for j in range(i+1, len(atom_rings)):
                ring1 = set(atom_rings[i])
                ring2 = set(atom_rings[j])
                if len(ring1.intersection(ring2)) >= 2:
                    # These rings are fused
                    # Find or create system
                    found = False
                    for system in fused_systems:
                        if i in system or j in system:
                            system.add(i)
                            system.add(j)
                            found = True
                            break
                    if not found:
                        fused_systems.append({i, j})
        
        # Merge overlapping systems
        merged = []
        for system in fused_systems:
            found = False
            for merged_system in merged:
                if system.intersection(merged_system):
                    merged_system.update(system)
                    found = True
                    break
            if not found:
                merged.append(system)
        
        return len(merged)

    def get_bridgehead_count(self, smiles: str) -> int:
        """Get count of bridgehead atoms."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return rdmd.CalcNumBridgeheadAtoms(mol)

    def _get_smallest_ring_size(self, smiles: str) -> int:
        """Get size of the smallest ring."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() == 0:
            return 0
        ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
        return min(ring_sizes)

    def _get_largest_ring_size(self, smiles: str) -> int:
        """Get size of the largest ring."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() == 0:
            return 0
        ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
        return max(ring_sizes)

    def get_smallest_or_largest_ring_count(self, smiles: str, 
                                          smallest: bool = True) -> int:
        """Get size of the smallest or largest ring."""
        if smallest:
            return self._get_smallest_ring_size(smiles)
        else:
            return self._get_largest_ring_size(smiles)

    def get_chain_termini_count(self, smiles: str) -> int:
        """Get count of chain termini (terminal carbons)."""
        indices = self.get_chain_termini_indices(smiles)
        return len(indices)

    def get_branch_point_count(self, smiles: str) -> int:
        """Get count of branch points."""
        indices = self.get_branch_point_indices(smiles)
        return len(indices)
    
    # - Indices tasks
    def get_ring_indices(self, smiles: str) -> List[int]:
        """
        Get indices of atoms that are part of any ring.

        Args:
            smiles: SMILES string

        Returns:
            List of atom indices that are in rings
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        ring_atoms = set()
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            ring_atoms.update(ring)

        return sorted(list(ring_atoms))

    def get_fused_ring_indices(self, smiles: str) -> List[int]:
        """
        Get atom indices of atoms in fused ring systems (rings sharing edges).

        Args:
            smiles: SMILES string

        Returns:
            List of atom indices in fused rings
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()
        indices = []

        # Find rings that share at least 2 atoms (edge fusion)
        for i, ring1 in enumerate(rings):
            for j, ring2 in enumerate(rings[i+1:], i+1):
                shared = set(ring1) & set(ring2)
                if len(shared) >= 2:  # Fused rings share at least 2 atoms
                    indices.extend(ring1)
                    indices.extend(ring2)

        return sorted(set(indices))
    
    def get_bridgehead_indices(self, smiles: str) -> List[int]:
        """
        Get atom indices of bridgehead atoms in the molecule.

        Definition follows RDKit's CalcNumBridgeheadAtoms (C++ source) so that
        indices and RDKit's count agree:
        - Iterate over all pairs of bond-rings (SSSR).
        - If the two rings share more than one bond, collect the atoms at the
          ends of those shared bonds in a counter.
        - Any atom with count == 1 across the shared-bond set for that pair is
          considered a bridgehead candidate for that pair.
        - The final set is the union of candidates across all ring pairs.

        Args:
            smiles: SMILES string

        Returns:
            List of atom indices that are bridgehead atoms
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        ri = mol.GetRingInfo()
        if ri.NumRings() == 0:
            return []

        bond_rings = list(ri.BondRings())
        if not bond_rings:
            return []

        bridgehead_set: set[int] = set()

        # Iterate pairs of rings
        for i in range(len(bond_rings)):
            ri_bonds = set(bond_rings[i])
            for j in range(i + 1, len(bond_rings)):
                rj_bonds = set(bond_rings[j])
                inter = ri_bonds & rj_bonds
                # Rings share at least two bonds
                if len(inter) > 1:
                    # Count per-atom occurrences across the shared bonds
                    atom_counts = [0] * mol.GetNumAtoms()
                    for bidx in inter:
                        b = mol.GetBondWithIdx(int(bidx))
                        atom_counts[b.GetBeginAtomIdx()] += 1
                        atom_counts[b.GetEndAtomIdx()] += 1
                    # Atoms with count == 1 are bridgehead candidates for this pair
                    for aidx, cnt in enumerate(atom_counts):
                        if cnt == 1:
                            bridgehead_set.add(aidx)

        return sorted(bridgehead_set)
    
    def _get_smallest_ring_indices(self, smiles: str) -> List[int]:
        """
        Get atom indices of the smallest ring(s) in the molecule.

        Args:
            smiles: SMILES string

        Returns:
            List of atom indices in the smallest ring(s)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() == 0:
            return []

        rings = ring_info.AtomRings()
        min_size = min(len(ring) for ring in rings)
        smallest_rings = [ring for ring in rings if len(ring) == min_size]

        # Combine all atoms from smallest rings and remove duplicates
        indices = sorted(set(atom for ring in smallest_rings for atom in ring))
        return indices
    
    def _get_largest_ring_indices(self, smiles: str) -> List[int]:
        """
        Get atom indices of the largest ring(s) in the molecule.

        If multiple rings have the same largest size, returns atoms from all of them.

        Args:
            smiles: SMILES string

        Returns:
            List of atom indices that belong to the largest ring(s)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        if not atom_rings:
            return []

        # Find the largest ring(s)
        max_size = max(len(ring) for ring in atom_rings)
        largest_rings = [ring for ring in atom_rings if len(ring) == max_size]

        # Combine all atoms from largest rings (remove duplicates)
        largest_ring_atoms = set()
        for ring in largest_rings:
            largest_ring_atoms.update(ring)

        return sorted(list(largest_ring_atoms))
    
    def get_smallest_or_largest_ring_indices(self, smiles: str, 
                                             smallest: bool = True) -> List[int]:
        """
        Get atom indices of the smallest or largest ring(s) in the molecule.
        
        Args:
            smiles: SMILES string
            smallest: If True, get smallest ring indices; if False, largest ring indices
            
        Returns:
            List of atom indices in the smallest or largest ring(s)
        """
        if smallest:
            return self._get_smallest_ring_indices(smiles)
        else:
            return self._get_largest_ring_indices(smiles)
    
    def get_chain_termini_indices(self, smiles: str) -> List[int]:
        """
        Get indices of terminal carbon atoms (degree 1) in chains.

        Args:
            smiles: SMILES string

        Returns:
            List of terminal carbon atom indices
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        return [i for i, atom in enumerate(mol.GetAtoms())
                if atom.GetDegree() == 1 and atom.GetSymbol() == 'C']

    def get_branch_point_indices(self, smiles: str) -> List[int]:
        """
        Get indices of carbon branch points (degree >= 3).

        Args:
            smiles: SMILES string

        Returns:
            List of branch point carbon atom indices
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        return [i for i, atom in enumerate(mol.GetAtoms())
                if atom.GetDegree() >= 3 and atom.GetSymbol() == 'C']
        #-----------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------
    # Chemistry-typed graph topology
    
    # - Count tasks
    def get_aromatic_ring_count(self, smiles: str) -> int:
        """Get count of aromatic rings."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return Descriptors.NumAromaticRings(mol)

    def get_aliphatic_ring_count(self, smiles: str) -> int:
        """Get count of aliphatic (non-aromatic) rings."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return Descriptors.NumAliphaticRings(mol)

    def get_heterocycle_count(self, smiles: str) -> int:
        """Get count of heterocyclic rings."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return rdmd.CalcNumHeterocycles(mol)

    def get_saturated_ring_count(self, smiles: str) -> int:
        """Get count of saturated rings."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return Descriptors.NumSaturatedRings(mol)
    
    def get_csp3_carbon_count(self, smiles: str) -> int:
        return len(self.get_csp3_carbon_indices(smiles))
    
    def get_longest_carbon_chain_count(self, smiles: str) -> int:
        """Get length of the longest continuous carbon chain."""
        indices = self.get_longest_carbon_chain_indices(smiles)
        return len(indices)
    
    def _get_stereocenter_indices_and_types(self, smiles: str) -> Tuple[List[int], Dict[int, str]]:
        """
        Get stereocenter atom indices and their R/S assignments.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Tuple of (atom_indices, stereocenter_types)
            - atom_indices: List of RDKit atom indices that are stereocenters
            - stereocenter_types: Dict mapping atom index -> R/S/unspecified assignment
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return [], {}
        
        # Use RDKit's FindMolChiralCenters for consistency with reward system
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        
        atom_indices = []
        stereocenter_types = {}
        
        for atom_idx, chirality in chiral_centers:
            atom_indices.append(atom_idx)
            stereocenter_types[atom_idx] = chirality if chirality in ['R', 'S'] else 'unspecified'
        
        return sorted(atom_indices), stereocenter_types

    def _get_r_stereocenter_count(self, smiles: str) -> int:
        """Get count of R-configured stereocenters."""
        _, types_dict = self._get_stereocenter_indices_and_types(smiles)
        return sum(1 for t in types_dict.values() if t == 'R')
    
    def _get_s_stereocenter_count(self, smiles: str) -> int:
        """Get count of S-configured stereocenters."""
        _, types_dict = self._get_stereocenter_indices_and_types(smiles)
        return sum(1 for t in types_dict.values() if t == 'S')
    
    def get_r_or_s_stereocenter_count(self, smiles: str, r_count: bool = True) -> int:
        if r_count:
            return self._get_r_stereocenter_count(smiles)
        else:
            return self._get_s_stereocenter_count(smiles)

    def get_unspecified_stereocenter_count(self, smiles: str) -> int:
        """Get count of unspecified/undefined stereocenters."""
        _, types_dict = self._get_stereocenter_indices_and_types(smiles)
        return sum(1 for t in types_dict.values() if t == 'unspecified')

    def get_stereocenter_count(self, smiles: str) -> int:
        """Get total count of stereocenters."""
        indices = self.get_stereocenter_indices(smiles)
        return len(indices)

    def _get_ez_double_bond_configurations(self, smiles: str) -> Tuple[List[List[int]], Dict[Tuple[int, int], str]]:
        """
        Get stereogenic double bond indices and their E/Z configurations.

        Args:
            smiles: SMILES string

        Returns:
            Tuple of (bond_indices, configurations)
            - bond_indices: List of [atom1_idx, atom2_idx] pairs
            - configurations: Dict mapping (atom1_idx, atom2_idx) -> "E"/"Z"/"unspecified"
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [], {}

        # Assign stereochemistry
        Chem.AssignStereochemistry(mol, cleanIt=False, force=True)

        # Find all potential stereo elements
        stereo_info = Chem.FindPotentialStereo(mol)

        bond_indices = []
        configurations = {}

        for element in stereo_info:
            # Only interested in double bond stereochemistry
            if element.type == Chem.StereoType.Bond_Double:
                # Get the bond
                bond = mol.GetBondWithIdx(element.centeredOn)
                atom1_idx = bond.GetBeginAtomIdx()
                atom2_idx = bond.GetEndAtomIdx()
                sorted_indices = sorted([atom1_idx, atom2_idx])
                bond_indices.append(sorted_indices)

                # Get configuration
                if element.specified == Chem.StereoSpecified.Specified:
                    stereo = bond.GetStereo()
                    if stereo == Chem.BondStereo.STEREOE:
                        configurations[tuple(sorted_indices)] = "E"
                    elif stereo == Chem.BondStereo.STEREOZ:
                        configurations[tuple(sorted_indices)] = "Z"
                    else:
                        configurations[tuple(sorted_indices)] = "unspecified"
                else:
                    configurations[tuple(sorted_indices)] = "unspecified"

        return bond_indices, configurations
    
    def _get_e_double_bond_count(self, smiles: str) -> int:
        """Get count of E-configured double bonds."""
        _, configurations = self._get_ez_double_bond_configurations(smiles)
        return sum(1 for config in configurations.values() if config == 'E')

    def _get_z_double_bond_count(self, smiles: str) -> int:
        """Get count of Z-configured double bonds."""
        _, configurations = self._get_ez_double_bond_configurations(smiles)
        return sum(1 for config in configurations.values() if config == 'Z')
    
    def get_e_z_stereochemistry_double_bond_count(self, smiles: str, e_count: bool = True) -> int:
        if e_count:
            return self._get_e_double_bond_count(smiles)
        else:
            return self._get_z_double_bond_count(smiles)

    def get_stereochemistry_unspecified_double_bond_count(self, smiles: str) -> int:
        """Get count of unspecified/unassigned stereogenic double bonds."""
        bonds = self._get_unspecified_double_bond_indices(smiles)
        return len(bonds)

    # - Indices tasks
    def get_aromatic_ring_indices(self, smiles: str) -> List[int]:
        """
        Get atom indices of atoms specifically in aromatic rings.
        Note: This is different from aromatic atoms, as it only includes atoms that are part of aromatic rings.

        Args:
            smiles: SMILES string

        Returns:
            List of atom indices in aromatic rings
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        ring_info = mol.GetRingInfo()
        indices = []

        for ring in ring_info.AtomRings():
            # Check if all atoms in ring are aromatic
            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                indices.extend(ring)

        return sorted(set(indices))

    def get_aliphatic_ring_indices(self, smiles: str) -> List[int]:
        """
        Get atom indices of atoms in aliphatic rings (per RDKit semantics).

        RDKit's NumAliphaticRings counts a ring as aliphatic if that ring has
        at least one non-aromatic bond (see RDKit source). We mirror that here
        for indices by including atoms from any ring that contains at least one
        non-aromatic bond.

        Args:
            smiles: SMILES string

        Returns:
            List of atom indices in aliphatic rings
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        ring_info = mol.GetRingInfo()
        indices: list[int] = []

        for ring in ring_info.AtomRings():
            # Determine if this ring has at least one non-aromatic bond
            has_non_aromatic_bond = False
            n = len(ring)
            for i in range(n):
                a1 = ring[i]
                a2 = ring[(i + 1) % n]
                b = mol.GetBondBetweenAtoms(a1, a2)
                if b is not None and not b.GetIsAromatic():
                    has_non_aromatic_bond = True
                    break
            if has_non_aromatic_bond:
                indices.extend(ring)

        return sorted(set(indices))
    
    def get_heterocycle_indices(self, smiles: str) -> List[int]:
        """
        Get atom indices of atoms in heterocyclic rings.

        Args:
            smiles: SMILES string

        Returns:
            List of atom indices in heterocycles
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        ring_info = mol.GetRingInfo()
        indices = []

        for ring in ring_info.AtomRings():
            # Check if ring contains heteroatom (non-C, non-H)
            if any(mol.GetAtomWithIdx(idx).GetSymbol() not in ['C', 'H'] for idx in ring):
                indices.extend(ring)

        return sorted(set(indices))

    def get_saturated_ring_indices(self, smiles: str) -> List[int]:
        """
        Get atom indices of atoms in saturated rings.
        A saturated ring has only single bonds (no double or triple bonds).

        Args:
            smiles: SMILES string

        Returns:
            List of atom indices in saturated rings
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        ring_info = mol.GetRingInfo()
        indices = []

        for ring in ring_info.AtomRings():
            # Check if ring is saturated (only single bonds between ring atoms)
            is_saturated = True

            # Check all bonds in the ring
            for i in range(len(ring)):
                atom1_idx = ring[i]
                atom2_idx = ring[(i + 1) % len(ring)]  # Next atom in ring (wraps around)

                # Get bond between these atoms
                bond = mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
                if bond and bond.GetBondType() != Chem.BondType.SINGLE:
                    is_saturated = False
                    break

            # Also check that atoms are not aromatic
            if is_saturated and not any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                indices.extend(ring)

        return sorted(set(indices))
    
    def get_csp3_carbon_indices(self, smiles: str) -> List[int]:
        """
        Get indices of sp3 hybridized carbon atoms.

        Args:
            smiles: SMILES string

        Returns:
            List of sp3 carbon atom indices
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        # SP3 carbons have 4 single bonds (tetrahedral)
        csp3_indices = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol() == 'C' and atom.GetHybridization() == Chem.HybridizationType.SP3:
                csp3_indices.append(i)

        return csp3_indices

    def get_longest_carbon_chain_indices(self, smiles: str) -> List[int]:
        """
        Get atom indices of the longest continuous carbon chain in the molecule.

        Default implementation uses the strict DFS-based method with a size guardrail
        (falls back to BFS diameter approximation for very large carbon graphs).

        Args:
            smiles: SMILES string

        Returns:
            List of atom indices that form the longest carbon chain
        """
        return self.get_strict_longest_carbon_chain_indices(smiles)

    def get_strict_longest_carbon_chain_indices(self, smiles: str,
                                                node_limit: int = 60) -> List[int]:
        """
        Compute exact (or near-exact) longest simple carbon chain indices using DFS
        with pruning on the carbon-only subgraph. Falls back to the BFS-based
        approximation when the carbon graph is larger than a size guardrail.

        Args:
            smiles: SMILES string
            node_limit: Max carbon nodes to attempt DFS (fallback above this)

        Returns:
            List of atom indices forming the longest carbon chain found
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        # Build carbon-only adjacency
        n_atoms = mol.GetNumAtoms()
        is_c = [mol.GetAtomWithIdx(i).GetSymbol() == 'C' for i in range(n_atoms)]
        carbon_nodes = [i for i, flag in enumerate(is_c) if flag]
        if not carbon_nodes:
            return []

        if len(carbon_nodes) > node_limit:
            # Too large for exact DFS within reasonable time; use approximation
            return self.get_longest_carbon_chain_indices(smiles)

        adj = {i: [] for i in carbon_nodes}
        for atom_idx in carbon_nodes:
            atom = mol.GetAtomWithIdx(atom_idx)
            for nbr in atom.GetNeighbors():
                j = nbr.GetIdx()
                if is_c[j]:
                    adj[atom_idx].append(j)

        # Order neighbors by descending degree to find long paths faster
        deg = {i: len(adj[i]) for i in adj}
        for i in adj:
            adj[i].sort(key=lambda x: deg[x], reverse=True)

        best_path: List[int] = []

        def dfs(u: int, visited: set, path: list):
            nonlocal best_path

            # Update best
            if len(path) > len(best_path):
                best_path = list(path)

            # Explore
            for v in adj.get(u, []):
                if v in visited:
                    continue
                visited.add(v)
                path.append(v)
                dfs(v, visited, path)
                path.pop()
                visited.remove(v)

        # Run DFS from each carbon node until time budget exhausted
        for s in carbon_nodes:
            dfs(s, {s}, [s])

        if best_path:
            return best_path
        # Fallback (e.g., degenerate graph)
        return self.get_longest_carbon_chain_indices(smiles)

    def get_strict_longest_carbon_chain_count(self, smiles: str) -> int:
        return len(self.get_strict_longest_carbon_chain_indices(smiles))


    def get_stereocenter_indices(self, smiles: str) -> List[int]:
        """
        Get just the RDKit atom indices of stereocenters.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of RDKit atom indices that are stereocenters
        """
        indices, _ = self._get_stereocenter_indices_and_types(smiles)
        return indices

    def _get_r_stereocenter_indices(self, smiles: str) -> List[int]:
        """
        Get indices of R stereocenters.

        Args:
            smiles: SMILES string

        Returns:
            List of R stereocenter atom indices
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        return [idx for idx, chirality in chiral_centers if chirality == 'R']

    def _get_s_stereocenter_indices(self, smiles: str) -> List[int]:
        """
        Get indices of S stereocenters.

        Args:
            smiles: SMILES string

        Returns:
            List of S stereocenter atom indices
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        return [idx for idx, chirality in chiral_centers if chirality == 'S']

    def get_r_or_s_stereocenter_indices(self, smiles: str, r_indices: bool = True) -> List[int]:
        if r_indices:
            return self._get_r_stereocenter_indices(smiles)
        else:
            return self._get_s_stereocenter_indices(smiles)
    
    def get_unspecified_stereocenter_indices(self, smiles: str) -> List[int]:
        """
        Get indices of unspecified stereocenters.

        Args:
            smiles: SMILES string

        Returns:
            List of unspecified stereocenter atom indices
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        return [idx for idx, chirality in chiral_centers if chirality == '?']
    
    def _get_e_double_bond_indices(self, smiles: str) -> List[List[int]]:
        """
        Get indices of E double bonds.

        Args:
            smiles: SMILES string

        Returns:
            List of [atom1_idx, atom2_idx] pairs for E double bonds
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        e_bonds = []
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                if bond.GetStereo() == Chem.BondStereo.STEREOE:
                    e_bonds.append(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))

        return e_bonds

    def _get_e_double_bond_indices_flat(self, smiles: str) -> List[int]:
        """
        Get flattened indices of atoms in E double bonds.
        Returns all unique atom indices involved in E-configured double bonds.

        Args:
            smiles: SMILES string

        Returns:
            Flat list of unique atom indices in E double bonds
        """
        nested = self._get_e_double_bond_indices(smiles)
        if not nested:
            return []
        # Flatten and get unique indices
        flat_indices = set()
        for bond in nested:
            flat_indices.update(bond)
        return sorted(list(flat_indices))

    def _get_z_double_bond_indices(self, smiles: str) -> List[List[int]]:
        """
        Get indices of Z double bonds.

        Args:
            smiles: SMILES string

        Returns:
            List of [atom1_idx, atom2_idx] pairs for Z double bonds
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        z_bonds = []
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                if bond.GetStereo() == Chem.BondStereo.STEREOZ:
                    z_bonds.append(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))

        return z_bonds
    
    def _get_unspecified_double_bond_indices(self, smiles: str) -> List[List[int]]:
        """
        Get indices of stereogenic double bonds with unspecified E/Z configuration.

        Args:
            smiles: SMILES string

        Returns:
            List of [atom1_idx, atom2_idx] pairs for unspecified double bonds
        """
        bond_indices, configurations = self._get_ez_double_bond_configurations(smiles)
        if not bond_indices:
            return []
        unspecified = []
        seen = set()
        for pair in bond_indices:
            key = tuple(sorted(pair))
            if key in seen:
                continue
            if configurations.get(key) == 'unspecified':
                unspecified.append(list(key))
                seen.add(key)
        return unspecified

    def _get_z_double_bond_indices_flat(self, smiles: str) -> List[int]:
        """
        Get flattened indices of atoms in Z double bonds.
        Returns all unique atom indices involved in Z-configured double bonds.

        Args:
            smiles: SMILES string

        Returns:
            Flat list of unique atom indices in Z double bonds
        """
        nested = self._get_z_double_bond_indices(smiles)
        if not nested:
            return []
        # Flatten and get unique indices
        flat_indices = set()
        for bond in nested:
            flat_indices.update(bond)
        return sorted(list(flat_indices))
    
    def get_e_z_stereochemistry_double_bond_indices(self, smiles: str, e_indices: bool = True) -> List[int]:
        if e_indices:
            return self._get_e_double_bond_indices_flat(smiles)
        else:
            return self._get_z_double_bond_indices_flat(smiles)
    
    def get_stereochemistry_unspecified_double_bond_indices(self, smiles: str) -> List[int]:
        """
        Get flattened indices of atoms in unspecified double bonds.
        Returns all unique atom indices involved in unspecified E/Z double bonds.

        Args:
            smiles: SMILES string

        Returns:
            Flat list of unique atom indices in unspecified double bonds
        """
        nested = self._get_unspecified_double_bond_indices(smiles)
        if not nested:
            return []
        # Flatten and get unique indices
        flat_indices = set()
        for bond in nested:
            flat_indices.update(bond)
        return sorted(list(flat_indices))
        
    #-----------------------------------------------------------------------------------
    # Composition

    # - Count tasks
    def get_hydrogen_count(self, smiles: str) -> int:
        """
        Get total count of hydrogen atoms (both explicit and implicit).

        Args:
            smiles: SMILES string

        Returns:
            Total number of hydrogen atoms
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0

        # Add implicit hydrogens to get total count
        mol = Chem.AddHs(mol)
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'H')

    def get_explicit_hydrogen_count(self, smiles: str) -> int:
        """
        Get count of explicitly shown hydrogen atoms in the SMILES.

        Args:
            smiles: SMILES string

        Returns:
            Number of explicit hydrogen atoms
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0

        # Count only explicit H atoms (those already in the molecule)
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'H')

    def get_carbon_count(self, smiles: str) -> int:
        return len(self.get_carbon_indices(smiles))
    
    def get_hetero_atom_count(self, smiles: str) -> int:
        """Get count of heteroatoms (non-C, non-H atoms)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() not in ['C', 'H'])
    
    def get_halogen_count(self, smiles: str) -> int:
        """Get count of halogen atoms (F, Cl, Br, I)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ['F', 'Cl', 'Br', 'I'])

    def get_heavy_atom_count(self, smiles: str) -> int:
        """Get count of heavy atoms (non-hydrogen atoms)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return mol.GetNumHeavyAtoms()
    
    def get_molecular_formula(self, smiles: str) -> str:
        """Get molecular formula."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return rdmd.CalcMolFormula(mol)

    # - Indices tasks
    def get_explicit_hydrogen_indices(self, smiles: str) -> List[int]:
        """
        Get indices of explicitly shown hydrogen atoms in the molecule.

        Args:
            smiles: SMILES string

        Returns:
            List of indices of explicit hydrogen atoms
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        # Get indices of explicit H atoms
        return [i for i, atom in enumerate(mol.GetAtoms())
                if atom.GetSymbol() == 'H']
    
    def get_carbon_indices(self, smiles: str) -> List[int]:
        """Get indices of all carbon atoms."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        return [i for i, atom in enumerate(mol.GetAtoms()) if atom.GetSymbol() == 'C']

    def get_hetero_atom_indices(self, smiles: str) -> List[int]:
        """
        Get indices of heteroatoms (non-carbon, non-hydrogen atoms) in the molecule.

        Args:
            smiles: SMILES string

        Returns:
            List of atom indices for heteroatoms
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        heteroatom_indices = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in ['C', 'H']:
                heteroatom_indices.append(atom.GetIdx())

        return sorted(heteroatom_indices)

    def get_halogen_indices(self, smiles: str) -> List[int]:
        """Get indices of all halogen atoms (F, Cl, Br, I)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        return [i for i, atom in enumerate(mol.GetAtoms()) if atom.GetSymbol() in ['F', 'Cl', 'Br', 'I']]

    def get_heavy_atom_indices(self, smiles: str) -> List[int]:
        """
        Get indices of heavy atoms (non-hydrogen atoms) in the molecule.

        Args:
            smiles: SMILES string

        Returns:
            List of atom indices for heavy atoms
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        indices = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol() != 'H':
                indices.append(i)

        return indices

    #-----------------------------------------------------------------------------------
    # Chemical perception
    
    # - Count tasks
    def get_hba_count(self, smiles: str) -> int:
        """Get count of hydrogen bond acceptors."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return Descriptors.NumHAcceptors(mol)

    def get_hbd_count(self, smiles: str) -> int:
        """Get count of hydrogen bond donors."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return Descriptors.NumHDonors(mol)

    def get_rotatable_bond_count(self, smiles: str) -> int:
        """Get count of rotatable bonds."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return Descriptors.NumRotatableBonds(mol)
    
    def get_oxidation_state_count(self, 
                                  smiles: str, 
                                  element:str, 
                                  max_oxidation: bool = True) -> int:
        indices = self.get_oxidation_state_indices(smiles, element, max_oxidation)
        return len(indices)

    # - Indices tasks
    def get_hba_indices(self, smiles: str) -> List[int]:
        """
        Get RDKit atom indices of hydrogen bond acceptors using RDKit's official SMARTS 
        pattern.
        Returns a flat list of unique atom indices that are hydrogen bond acceptors.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of unique atom indices that are hydrogen bond acceptors
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []
        
        # RDKit's official HBA SMARTS pattern
        hba_pattern = ("[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
                       "$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]")
        
        pat = Chem.MolFromSmarts(hba_pattern)
        if not pat:
            return []
        
        # Find all matches - each match is a tuple of atom indices
        matches = mol.GetSubstructMatches(pat)
        
        # Flatten and get unique indices
        hba_indices = set()
        for match in matches:
            hba_indices.update(match)
        
        return sorted(list(hba_indices))
    
    def get_hbd_indices(self, smiles: str) -> List[int]:
        """
        Get RDKit atom indices of hydrogen bond donors using RDKit's official SMARTS 
        pattern.
        Returns a flat list of unique atom indices that are hydrogen bond donors.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of unique atom indices that are hydrogen bond donors
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []
        
        # RDKit's official HBD SMARTS pattern
        hbd_pattern = "[N&!H0&v3,N&!H0&+1&v4,O&H1&+0,S&H1&+0,n&H1&+0]"
        
        pat = Chem.MolFromSmarts(hbd_pattern)
        if not pat:
            return []
        
        # Find all matches - each match is a tuple of atom indices
        matches = mol.GetSubstructMatches(pat)
        
        # Flatten and get unique indices
        hbd_indices = set()
        for match in matches:
            hbd_indices.update(match)
        
        return sorted(list(hbd_indices))
    
    def get_rotatable_bond_indices(self, smiles: str) -> List[int]:
        """
        Get atom indices connected by rotatable bonds.

        Note: This returns the atoms involved in rotatable bonds. Since rotatable
        bonds can share atoms (e.g., C-C-C has 2 bonds but 3 atoms), the number
        of indices may not be exactly 2 * number of bonds.

        Args:
            smiles: SMILES string

        Returns:
            List of unique atom indices that are part of rotatable bonds
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        atom_indices: set[int] = set()
        num_rotatable = rdmd.CalcNumRotatableBonds(mol)

        if num_rotatable == 0:
            return []

        matches = ()
        if STRICT_ROTATABLE_BOND_PATTERN is not None:
            matches = mol.GetSubstructMatches(
                STRICT_ROTATABLE_BOND_PATTERN, uniquify=True
            )

        if len(matches) != num_rotatable:
            matches = mol.GetSubstructMatches(
                Lipinski.RotatableBondSmarts, uniquify=True
            )

        # Fallback: if RDKit still reports a different count, trust RDKit for the total
        # but only include as many matches as observed.
        for match in matches[:num_rotatable]:
            atom_indices.update(match)

        return sorted(atom_indices)
    
    def _get_max_oxidation_C_indices(self, smiles: str) -> List[int]:
        """
        Get indices of carbon atoms at their maximum oxidation state.

        Args:
            smiles: SMILES string

        Returns:
            List of carbon atom indices at maximum oxidation state
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        # Calculate oxidation numbers
        AllChem.CalcOxidationNumbers(mol)

        # First pass: collect all carbon oxidation states
        carbon_ox_states = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                carbon_ox_states.append(ox_state)

        if not carbon_ox_states:
            return []

        max_ox = max(carbon_ox_states)

        # Second pass: get indices of carbons at max oxidation
        indices = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                if ox_state == max_ox:
                    indices.append(atom.GetIdx())

        return indices

    def _get_min_oxidation_C_indices(self, smiles: str) -> List[int]:
        """
        Get indices of carbon atoms at their minimum oxidation state.

        Args:
            smiles: SMILES string

        Returns:
            List of carbon atom indices at minimum oxidation state
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        # Calculate oxidation numbers
        AllChem.CalcOxidationNumbers(mol)

        # First pass: collect all carbon oxidation states
        carbon_ox_states = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                carbon_ox_states.append(ox_state)

        if not carbon_ox_states:
            return []

        min_ox = min(carbon_ox_states)

        # Second pass: get indices of carbons at min oxidation
        indices = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                if ox_state == min_ox:
                    indices.append(atom.GetIdx())

        return indices

    def _get_max_oxidation_N_indices(self, smiles: str) -> List[int]:
        """
        Get indices of nitrogen atoms at their maximum oxidation state.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of nitrogen atom indices at maximum oxidation state
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        # Calculate oxidation numbers
        AllChem.CalcOxidationNumbers(mol)
        
        # First pass: collect all nitrogen oxidation states
        nitrogen_ox_states = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                nitrogen_ox_states.append(ox_state)
        
        if not nitrogen_ox_states:
            return []
        
        max_ox = max(nitrogen_ox_states)
        
        # Second pass: get indices of nitrogens at max oxidation
        indices = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                if ox_state == max_ox:
                    indices.append(atom.GetIdx())
        
        return indices
        
    
    def _get_min_oxidation_N_indices(self, smiles: str) -> List[int]:
        """
        Get indices of nitrogen atoms at their minimum oxidation state.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of nitrogen atom indices at minimum oxidation state
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        # Calculate oxidation numbers
        AllChem.CalcOxidationNumbers(mol)
        
        # First pass: collect all nitrogen oxidation states
        nitrogen_ox_states = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                nitrogen_ox_states.append(ox_state)
        
        if not nitrogen_ox_states:
            return []
        
        min_ox = min(nitrogen_ox_states)
        
        # Second pass: get indices of nitrogens at min oxidation
        indices = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                if ox_state == min_ox:
                    indices.append(atom.GetIdx())
        
        return indices
        
    
    def _get_max_oxidation_O_indices(self, smiles: str) -> List[int]:
        """
        Get indices of oxygen atoms at their maximum oxidation state.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of oxygen atom indices at maximum oxidation state
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        # Calculate oxidation numbers
        AllChem.CalcOxidationNumbers(mol)
        
        # First pass: collect all oxygen oxidation states
        oxygen_ox_states = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'O' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                oxygen_ox_states.append(ox_state)
        
        if not oxygen_ox_states:
            return []
        
        max_ox = max(oxygen_ox_states)
        
        # Second pass: get indices of oxygens at max oxidation
        indices = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'O' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                if ox_state == max_ox:
                    indices.append(atom.GetIdx())
        
        return indices
        
    
    def _get_min_oxidation_O_indices(self, smiles: str) -> List[int]:
        """
        Get indices of oxygen atoms at their minimum oxidation state.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of oxygen atom indices at minimum oxidation state
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        # Calculate oxidation numbers
        AllChem.CalcOxidationNumbers(mol)
        
        # First pass: collect all oxygen oxidation states
        oxygen_ox_states = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'O' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                oxygen_ox_states.append(ox_state)
        
        if not oxygen_ox_states:
            return []
        
        min_ox = min(oxygen_ox_states)
        
        # Second pass: get indices of oxygens at min oxidation
        indices = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'O' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                if ox_state == min_ox:
                    indices.append(atom.GetIdx())
        
        return indices
        
    
    def _get_max_oxidation_P_indices(self, smiles: str) -> List[int]:
        """
        Get indices of phosphorus atoms at their maximum oxidation state.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of phosphorus atom indices at maximum oxidation state
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        # Calculate oxidation numbers
        AllChem.CalcOxidationNumbers(mol)
        
        # First pass: collect all phosphorus oxidation states
        phosphorus_ox_states = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'P' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                phosphorus_ox_states.append(ox_state)
        
        if not phosphorus_ox_states:
            return []
        
        max_ox = max(phosphorus_ox_states)
        
        # Second pass: get indices of phosphorus at max oxidation
        indices = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'P' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                if ox_state == max_ox:
                    indices.append(atom.GetIdx())
        
        return indices
        
    
    def _get_min_oxidation_P_indices(self, smiles: str) -> List[int]:
        """
        Get indices of phosphorus atoms at their minimum oxidation state.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of phosphorus atom indices at minimum oxidation state
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        # Calculate oxidation numbers
        AllChem.CalcOxidationNumbers(mol)
        
        # First pass: collect all phosphorus oxidation states
        phosphorus_ox_states = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'P' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                phosphorus_ox_states.append(ox_state)
        
        if not phosphorus_ox_states:
            return []
        
        min_ox = min(phosphorus_ox_states)
        
        # Second pass: get indices of phosphorus at min oxidation
        indices = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'P' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                if ox_state == min_ox:
                    indices.append(atom.GetIdx())
        
        return indices
        
    
    def _get_max_oxidation_S_indices(self, smiles: str) -> List[int]:
        """
        Get indices of sulfur atoms at their maximum oxidation state.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of sulfur atom indices at maximum oxidation state
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        # Calculate oxidation numbers
        AllChem.CalcOxidationNumbers(mol)
        
        # First pass: collect all sulfur oxidation states
        sulfur_ox_states = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'S' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                sulfur_ox_states.append(ox_state)
        
        if not sulfur_ox_states:
            return []
        
        max_ox = max(sulfur_ox_states)
        
        # Second pass: get indices of sulfur at max oxidation
        indices = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'S' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                if ox_state == max_ox:
                    indices.append(atom.GetIdx())
        
        return indices
        
    
    def _get_min_oxidation_S_indices(self, smiles: str) -> List[int]:
        """
        Get indices of sulfur atoms at their minimum oxidation state.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of sulfur atom indices at minimum oxidation state
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        # Calculate oxidation numbers
        AllChem.CalcOxidationNumbers(mol)
        
        # First pass: collect all sulfur oxidation states
        sulfur_ox_states = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'S' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                sulfur_ox_states.append(ox_state)
        
        if not sulfur_ox_states:
            return []
        
        min_ox = min(sulfur_ox_states)
        
        # Second pass: get indices of sulfur at min oxidation
        indices = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'S' and atom.HasProp('OxidationNumber'):
                ox_state = int(atom.GetProp('OxidationNumber'))
                if ox_state == min_ox:
                    indices.append(atom.GetIdx())
        
        return indices
        
    
    def get_oxidation_state_indices(self, smiles:str, element:str, max_oxidation: bool = True) -> List[int]:
        if max_oxidation:
            if element == 'C':
                indices = self._get_max_oxidation_C_indices(smiles)
            elif element == 'N':
                indices = self._get_max_oxidation_N_indices(smiles)
            elif element == 'O':
                indices = self._get_max_oxidation_O_indices(smiles)
            elif element == 'P':
                indices = self._get_max_oxidation_P_indices(smiles)
            elif element == 'S':
                indices = self._get_max_oxidation_S_indices(smiles)
            else:
                raise ValueError(f"Element '{element}' not supported for oxidation state tasks.")
        else:
            if element == 'C':
                indices = self._get_min_oxidation_C_indices(smiles)
            elif element == 'N':
                indices = self._get_min_oxidation_N_indices(smiles)
            elif element == 'O':
                indices = self._get_min_oxidation_O_indices(smiles)
            elif element == 'P':
                indices = self._get_min_oxidation_P_indices(smiles)
            elif element == 'S':
                indices = self._get_min_oxidation_S_indices(smiles)
            else:
                raise ValueError(f"Element '{element}' not supported for oxidation state tasks.")

        return indices
    
    
    #-----------------------------------------------------------------------------------
    # Functional groups
    def get_functional_group_count_and_indices(self, smiles) -> dict:
        fg_dict = self.functional_group_solver.get_counts_and_indices(smiles)
        return fg_dict

    #-----------------------------------------------------------------------------------
    # Synthesis

    # - Count tasks
    def get_brics_fragment_count(self, smiles: str) -> int:
        """
        Get count of BRICS fragments when molecule is decomposed.

        Args:
            smiles: SMILES string

        Returns:
            Number of BRICS fragments
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0

        # Perform BRICS decomposition
        fragments = BRICS.BRICSDecompose(mol)
        return len(fragments)

    # - Indices tasks
    def get_brics_bond_indices(self, smiles: str) -> List[int]:
        """
        Get atom indices where BRICS bonds would be broken.
        Returns the atom indices that are at the ends of breakable bonds.

        Args:
            smiles: SMILES string

        Returns:
            List of atom indices at BRICS breakable bond positions
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        # Find bonds that match BRICS breaking rules
        bonds = BRICS.FindBRICSBonds(mol)

        # Extract unique atom indices from the bond tuples
        atom_indices = set()
        for bond_info in bonds:
            # bond_info is ((atom1, atom2), (type1, type2))
            bond_atoms = bond_info[0]
            atom_indices.add(bond_atoms[0])
            atom_indices.add(bond_atoms[1])

        return sorted(list(atom_indices))

    # - Template based reactions
    def get_reaction_counts_and_indices(self, smiles: str) -> dict:
        reaction_dict = self.reaction_solver.get_reaction_data(smiles)
        return reaction_dict

    # Murcko Scaffold

    def get_murcko_scaffold_count(self, smiles: str) -> int:
        """
        Get the number of atoms in the Murcko scaffold.
        Returns 0 if there is no scaffold (e.g., for linear molecules).

        Args:
            smiles: SMILES string

        Returns:
            Number of atoms in the Murcko scaffold
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0

        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)

        if scaffold is None:
            return 0

        return scaffold.GetNumAtoms()

    def get_murcko_scaffold_indices(self, smiles: str) -> List[int]:
        """
        Get the atom indices that are part of the Murcko scaffold.
        Returns empty list if there is no scaffold.

        Args:
            smiles: SMILES string

        Returns:
            List of atom indices in the Murcko scaffold
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        from rdkit.Chem.Scaffolds import MurckoScaffold

        # Get the scaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return []

        # Find matching atoms between original molecule and scaffold
        # We need to do a substructure match to map scaffold atoms to original indices
        match = mol.GetSubstructMatch(scaffold)

        if not match:
            # If direct match fails, try to find the scaffold atoms differently
            # Get the scaffold SMILES and match it back
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
            if scaffold_mol:
                match = mol.GetSubstructMatch(scaffold_mol)

        return sorted(list(match)) if match else []

    def get_murcko_scaffold_value(self, smiles: str) -> str:
        """
        Get the SMILES string of the Murcko scaffold.
        Returns empty string if there is no scaffold.

        Args:
            smiles: SMILES string

        Returns:
            SMILES string of the Murcko scaffold
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""

        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)

        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return ""

        return Chem.MolToSmiles(scaffold)

    def get_spiro_count(self, smiles: str) -> int:
        """Get the number of spiro atoms."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return rdmd.CalcNumSpiroAtoms(mol)

    def get_spiro_indices(self, smiles: str) -> List[int]:
        """Get indices of spiro atoms (atoms shared by exactly two rings)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        ring_info = mol.GetRingInfo()
        atom_rings = [set(ring) for ring in ring_info.AtomRings()]

        if len(atom_rings) < 2:
            return []

        spiro_indices = []
        for atom_idx in range(mol.GetNumAtoms()):
            rings_containing_atom = [ring for ring in atom_rings if atom_idx in ring]
            if len(rings_containing_atom) != 2:
                continue

            shared_atoms = rings_containing_atom[0].intersection(rings_containing_atom[1])
            if shared_atoms == {atom_idx}:
                spiro_indices.append(atom_idx)

        return sorted(spiro_indices)

    # Alias methods for consistency
    def get_carbon_atom_count(self, smiles: str) -> int:
        """Alias for get_carbon_count for consistency."""
        return self.get_carbon_count(smiles)

    def get_carbon_atom_indices(self, smiles: str) -> List[int]:
        """Alias for get_carbon_indices for consistency."""
        return self.get_carbon_indices(smiles)

    # Note: get_hetero_atom_count already exists at line 1022
    # Note: get_hetero_atom_indices already exists at line 1079

    def get_halogen_atom_count(self, smiles: str) -> int:
        """Alias for get_halogen_count for consistency."""
        return self.get_halogen_count(smiles)

    def get_halogen_atom_indices(self, smiles: str) -> List[int]:
        """Alias for get_halogen_indices for consistency."""
        return self.get_halogen_indices(smiles)

    # ============================================================================
    # Additional aliases for complete column_category_map.py compatibility
    # ============================================================================

    def get_smallest_largest_ring_size_smallest_count(self, smiles: str) -> int:
        """Alias for compatibility with column names"""
        return self.get_smallest_or_largest_ring_count(smiles, smallest=True)

    def get_smallest_largest_ring_size_largest_count(self, smiles: str) -> int:
        """Alias for compatibility with column names"""
        return self.get_smallest_or_largest_ring_count(smiles, smallest=False)

    def get_smallest_largest_ring_size_smallest_index(self, smiles: str) -> List[int]:
        """Alias for compatibility with column names"""
        return self.get_smallest_or_largest_ring_indices(smiles, smallest=True)

    def get_smallest_largest_ring_size_largest_index(self, smiles: str) -> List[int]:
        """Alias for compatibility with column names"""
        return self.get_smallest_or_largest_ring_indices(smiles, smallest=False)

    def get_hydrogen_atom_count(self, smiles: str) -> int:
        """Alias for get_hydrogen_count for consistency"""
        return self.get_hydrogen_count(smiles)

    def get_smallest_or_largest_ring_size(self, smiles: str, smallest: bool = True) -> int:
        """Alias for get_smallest_or_largest_ring_count"""
        return self.get_smallest_or_largest_ring_count(smiles, smallest=smallest)

#---------------------------------------------------------------------------------------
# Debugging
