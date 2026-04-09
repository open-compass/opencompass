"""
Solver for functional group tasks:
The solver identifies functional groups provided as SMARTS patterns in a given molecule
and returns count and index values.
"""

#---------------------------------------------------------------------------------------
# Config
from dataclasses import dataclass
from pathlib import Path
from .._data import SMARTS_FUNCTIONAL_GROUPS

@dataclass
class FunctionalGroupSolverConfig:
    smarts_patterns_path: Path = SMARTS_FUNCTIONAL_GROUPS

#---------------------------------------------------------------------------------------
# Imports
from typing import List, Tuple, Dict
from rdkit import Chem
#---------------------------------------------------------------------------------------
# Class definitions

class FunctionalGroupSolver:
    def __init__(self, 
                 config: FunctionalGroupSolverConfig=FunctionalGroupSolverConfig()):
        self.config = config
        
        # Initialize and load functional groups
        (self.smarts_patterns, self.names
         ) = self._load_smarts_patterns(config.smarts_patterns_path)
        
        # Pre-compile SMARTS patterns
        self.compiled_patterns = {}
        self._compile_patterns()

        # Procedural detectors for substituent-accurate groups
        self.procedural_detectors = {
            'ethyl': self._detect_ethyl_substituents,
            'allyl': self._detect_allyl_substituents,
            'propargyl': self._detect_propargyl_substituents,
            'benzyl': self._detect_benzyl_substituents,
            'isopropyl': self._detect_isopropyl_substituents,
            'isobutyl': self._detect_isobutyl_substituents,
            'sec_butyl': self._detect_sec_butyl_substituents,
            'tert_butyl': self._detect_tert_butyl_substituents,
        }
        
    def get_counts_and_indices(self, smiles: str) -> Dict[str, Dict[str, any]]:
        """
        Get both counts and indices for all functional groups.
        """
        results = {}
        
        # Parse molecule once
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {name: {'indices': [], 'count': 0, 'instances': 0} 
                    for name in self.compiled_patterns}
        
        # Process all patterns
        # Iterate all known names (from SMARTS file); if a procedural detector
        # exists, it overrides SMARTS-based detection for that name
        for fg_name, patterns in self.compiled_patterns.items():
            atom_indices = set()
            total_instances = 0

            if fg_name in self.procedural_detectors:
                instances = self.procedural_detectors[fg_name](mol)
                total_instances = len(instances)
                for inst in instances:
                    for idx in inst:
                        atom_indices.add(idx)
            else:
                # Check each pattern (some FGs have multiple SMARTS)
                for pattern in patterns:
                    matches = mol.GetSubstructMatches(pattern)
                    total_instances += len(matches)
                    for match in matches:
                        atom_indices.update(match)

            indices_list = sorted(list(atom_indices))
            results[f"functional_group_{fg_name}_count"] = len(indices_list)
            results[f"functional_group_{fg_name}_index"] = indices_list
            results[f"functional_group_{fg_name}_nbrInstances"] = total_instances
        
        return results

    #-----------------------------------------------------------------------------------
    # General utility methods
    def _load_smarts_patterns(self, smarts_file: str):
        """Load SMARTS patterns."""

        smarts_patterns = {}
        names = []

        with open(smarts_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(':')
                    if len(parts) >= 3:
                        name = parts[0].strip()
                        smarts = parts[2].strip()
                        smarts_patterns[name] = smarts
                        names.append(name)

        return smarts_patterns, names

    def get_functional_groups(self) -> Tuple[dict, List[str]]:
        """Return the a dictionary of functional groups and a list of functional group 
        names."""
        return self.smarts_patterns, self.names

    def _compile_patterns(self):
        """Pre-compile all SMARTS patterns for efficiency."""

        for name, smarts_str in self.smarts_patterns.items():
            # Compile SMARTS string to Mol object
            if smarts_str is not None:
                pattern_mol = Chem.MolFromSmarts(smarts_str)
                if pattern_mol is not None:
                    self.compiled_patterns[name] = [pattern_mol]  # Store as list for consistency
                else:
                    print(f"Warning: Could not compile SMARTS for {name}: {smarts_str}")
                    self.compiled_patterns[name] = []

    # ---------------------------------------------------------------------
    # Procedural substituent detectors
    def _detect_ethyl_substituents(self, mol) -> list:
        """
        Detect true ethyl substituents: R-CH2-CH3
        Returns list of instances, each as a list of atom indices [CH3, CH2].
        """
        instances = []
        for a in mol.GetAtoms():
            if a.GetSymbol() != 'C':
                continue
            # CH3 end: one heavy neighbor, many Hs
            if a.GetDegree() != 1 or a.GetTotalNumHs() < 3:
                continue
            nbr = a.GetNeighbors()[0]
            if nbr.GetSymbol() != 'C':
                continue
            # CH2 attachment carbon: at least two heavy neighbors (one is CH3, one is R)
            if nbr.GetTotalNumHs() < 2:
                continue
            if nbr.GetDegree() < 2:
                continue
            # Require that the "attachment" neighbor (other than CH3) is not
            # just a linear sp3 carbon (to avoid matching the ethyl tail of a
            # longer alkyl like n-propyl). Accept anchors that are heteroatoms
            # or non-sp3 carbons or branched/ring/aryl carbons.
            ok = False
            for anchor in nbr.GetNeighbors():
                if anchor.GetIdx() == a.GetIdx():
                    continue
                if anchor.GetSymbol() != 'C':
                    ok = True
                    break
                # For carbon anchors: accept if not a linear sp3 CH2
                if (anchor.GetHybridization() != Chem.HybridizationType.SP3
                    or anchor.GetDegree() != 2
                    or anchor.GetIsAromatic()
                    or anchor.IsInRing()):
                    ok = True
                    break
            if ok:
                instances.append([a.GetIdx(), nbr.GetIdx()])
        return instances

    def _detect_allyl_substituents(self, mol) -> list:
        """
        Detect true allyl substituents: R-CH2-CH=CH2
        Returns list of instances [CH2_alkyl, vinylic_C1, vinylic_CH2].
        """
        instances = []
        for ch2 in mol.GetAtoms():
            if ch2.GetSymbol() != 'C' or ch2.GetTotalNumHs() < 2:
                continue
            # Need at least two heavy neighbors (one is vinylic C, one is R)
            if ch2.GetDegree() < 2:
                continue
            for v1 in ch2.GetNeighbors():
                if v1.GetSymbol() != 'C':
                    continue
                bond = mol.GetBondBetweenAtoms(ch2.GetIdx(), v1.GetIdx())
                if not bond or bond.GetBondType() != Chem.BondType.SINGLE:
                    continue
                # v1 must be double-bonded to v2 which is terminal CH2
                for v1_nb in v1.GetNeighbors():
                    if v1_nb.GetIdx() == ch2.GetIdx():
                        continue
                    if v1_nb.GetSymbol() != 'C':
                        continue
                    b12 = mol.GetBondBetweenAtoms(v1.GetIdx(), v1_nb.GetIdx())
                    if not b12 or b12.GetBondType() != Chem.BondType.DOUBLE:
                        continue
                    # terminal vinyl CH2
                    if v1_nb.GetTotalNumHs() >= 2 and v1_nb.GetDegree() == 1:
                        instances.append([ch2.GetIdx(), v1.GetIdx(), v1_nb.GetIdx()])
        return instances

    def _detect_propargyl_substituents(self, mol) -> list:
        """
        Detect true propargyl substituents: R-CH2-C#CH
        Returns list of instances [CH2_alkyl, sp_C1, terminal_alkynyl_C].
        """
        instances = []
        for ch2 in mol.GetAtoms():
            if ch2.GetSymbol() != 'C' or ch2.GetTotalNumHs() < 2:
                continue
            if ch2.GetDegree() < 2:
                continue
            for sp1 in ch2.GetNeighbors():
                if sp1.GetSymbol() != 'C':
                    continue
                b = mol.GetBondBetweenAtoms(ch2.GetIdx(), sp1.GetIdx())
                if not b or b.GetBondType() != Chem.BondType.SINGLE:
                    continue
                # Look for sp1#sp2 with terminal CH
                for sp2 in sp1.GetNeighbors():
                    if sp2.GetIdx() == ch2.GetIdx():
                        continue
                    if sp2.GetSymbol() != 'C':
                        continue
                    b12 = mol.GetBondBetweenAtoms(sp1.GetIdx(), sp2.GetIdx())
                    if not b12 or b12.GetBondType() != Chem.BondType.TRIPLE:
                        continue
                    # terminal alkyne CH (at least one H, single heavy neighbor)
                    if sp2.GetTotalNumHs() >= 1 and sp2.GetDegree() == 1:
                        instances.append([ch2.GetIdx(), sp1.GetIdx(), sp2.GetIdx()])
        return instances

    def _detect_benzyl_substituents(self, mol) -> list:
        """
        Detect benzyl substituents: R-CH2-Ph
        Returns list of instances [aryl_C, CH2].
        """
        instances = []
        for ch2 in mol.GetAtoms():
            if ch2.GetSymbol() != 'C' or ch2.GetTotalNumHs() < 2:
                continue
            # Must have an aromatic carbon neighbor
            for nb in ch2.GetNeighbors():
                if nb.GetIsAromatic() and nb.GetSymbol() == 'C':
                    instances.append([nb.GetIdx(), ch2.GetIdx()])
                    break
        return instances

    def _detect_isopropyl_substituents(self, mol) -> list:
        """
        Detect isopropyl substituents: R-CH(CH3)2
        Return instances [central_C, methyl1_C, methyl2_C].
        """
        instances = []
        for c in mol.GetAtoms():
            if c.GetSymbol() != 'C':
                continue
            if c.GetHybridization() != Chem.HybridizationType.SP3:
                continue
            if c.GetDegree() != 3:
                continue
            methyls = []
            anchor = None
            for nb in c.GetNeighbors():
                if nb.GetSymbol() != 'C':
                    anchor = nb
                else:
                    if nb.GetDegree() == 1 and nb.GetTotalNumHs() >= 3:
                        methyls.append(nb)
                    else:
                        anchor = nb if anchor is None else anchor
            if len(methyls) == 2 and anchor is not None:
                instances.append([c.GetIdx(), methyls[0].GetIdx(), methyls[1].GetIdx()])
        return instances

    def _detect_tert_butyl_substituents(self, mol) -> list:
        """
        Detect tert-butyl substituents: R-C(CH3)3
        Return instances [central_C, m1, m2, m3].
        """
        instances = []
        for c in mol.GetAtoms():
            if c.GetSymbol() != 'C':
                continue
            if c.GetHybridization() != Chem.HybridizationType.SP3:
                continue
            if c.GetDegree() != 4:
                continue
            methyls = []
            anchors = []
            for nb in c.GetNeighbors():
                if nb.GetSymbol() == 'C' and nb.GetDegree() == 1 and nb.GetTotalNumHs() >= 3:
                    methyls.append(nb)
                else:
                    anchors.append(nb)
            if len(methyls) == 3 and len(anchors) == 1:
                instances.append([c.GetIdx()] + [m.GetIdx() for m in methyls])
        return instances

    def _detect_isobutyl_substituents(self, mol) -> list:
        """
        Detect isobutyl substituents: R-CH2-CH(CH3)2
        Return instances [CH2_attach, central_CH, methyl1, methyl2].
        """
        instances = []
        for ch2 in mol.GetAtoms():
            if ch2.GetSymbol() != 'C':
                continue
            if ch2.GetHybridization() != Chem.HybridizationType.SP3:
                continue
            # Attachment CH2 must have at least two heavy neighbors
            if ch2.GetDegree() < 2 or ch2.GetTotalNumHs() < 2:
                continue
            for center in ch2.GetNeighbors():
                if center.GetSymbol() != 'C':
                    continue
                if center.GetHybridization() != Chem.HybridizationType.SP3:
                    continue
                if center.GetDegree() != 3:
                    continue
                # center must have two methyl neighbors (besides ch2)
                methyls = []
                ok = False
                for nb in center.GetNeighbors():
                    if nb.GetIdx() == ch2.GetIdx():
                        ok = True
                        continue
                    if nb.GetSymbol() == 'C' and nb.GetDegree() == 1 and nb.GetTotalNumHs() >= 3:
                        methyls.append(nb)
                if ok and len(methyls) == 2:
                    instances.append([ch2.GetIdx(), center.GetIdx(), methyls[0].GetIdx(), methyls[1].GetIdx()])
        return instances

    def _detect_sec_butyl_substituents(self, mol) -> list:
        """
        Detect sec-butyl substituents: R-CH(CH3)-CH2-CH3
        Return instances [sec_C, methyl, CH2, terminal_CH3].
        """
        instances = []
        for sec in mol.GetAtoms():
            if sec.GetSymbol() != 'C':
                continue
            if sec.GetHybridization() != Chem.HybridizationType.SP3:
                continue
            if sec.GetDegree() != 3:
                continue
            # Identify neighbors
            carbon_nbs = [nb for nb in sec.GetNeighbors() if nb.GetSymbol() == 'C']
            if len(carbon_nbs) < 2:
                continue
            methyl = None
            chain = None
            for nb in carbon_nbs:
                if nb.GetDegree() == 1 and nb.GetTotalNumHs() >= 3:
                    methyl = nb
                else:
                    chain = nb
            if methyl is None or chain is None:
                continue
            # chain should be CH2 connected to terminal CH3
            if chain.GetHybridization() != Chem.HybridizationType.SP3:
                continue
            if chain.GetDegree() != 2 or chain.GetTotalNumHs() < 2:
                continue
            term = None
            for nb2 in chain.GetNeighbors():
                if nb2.GetIdx() == sec.GetIdx():
                    continue
                if nb2.GetSymbol() == 'C' and nb2.GetDegree() == 1 and nb2.GetTotalNumHs() >= 3:
                    term = nb2
                    break
            if term is None:
                continue
            instances.append([sec.GetIdx(), methyl.GetIdx(), chain.GetIdx(), term.GetIdx()])
        return instances
    

