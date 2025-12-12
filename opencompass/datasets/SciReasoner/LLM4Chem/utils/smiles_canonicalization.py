from rdchiral.chiral import copy_chirality
from rdkit import Chem, RDLogger
from rdkit.Chem.AllChem import AssignStereochemistry

RDLogger.DisableLog('rdApp.*')


def canonicalize(smiles, isomeric=False, canonical=True, kekulize=False):
    # When canonicalizing a SMILES string, we typically want to
    # run Chem.RemoveHs(mol), but this will try to kekulize the mol
    # which is not required for canonical SMILES.  Instead, we make a
    # copy of the mol retaining only the information we desire
    # (not explicit Hs)
    # Then, we sanitize the mol without kekulization.
    # copy_atom and copy_edit_mol
    # Are used to create this clean copy of the mol.
    def copy_atom(atom):
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        if atom.GetIsAromatic() and atom.GetNoImplicit():
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
            # elif atom.GetSymbol() == 'N':
            #    print(atom.GetSymbol())
            #    print(atom.GetImplicitValence())
            #    new_atom.SetNumExplicitHs(-atom.GetImplicitValence())
            # elif atom.GetSymbol() == 'S':
            #    print(atom.GetSymbol())
            #    print(atom.GetImplicitValence())
        return new_atom

    def copy_edit_mol(mol):
        new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
        for atom in mol.GetAtoms():
            new_atom = copy_atom(atom)
            new_mol.AddAtom(new_atom)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            new_mol.AddBond(a1, a2, bt)
            new_bond = new_mol.GetBondBetweenAtoms(a1, a2)
            new_bond.SetBondDir(bond.GetBondDir())
            new_bond.SetStereo(bond.GetStereo())
        for new_atom in new_mol.GetAtoms():
            atom = mol.GetAtomWithIdx(new_atom.GetIdx())
            copy_chirality(atom, new_atom)
        return new_mol

    smiles = smiles.replace(' ', '')
    tmp = Chem.MolFromSmiles(smiles, sanitize=False)
    tmp.UpdatePropertyCache()
    new_mol = copy_edit_mol(tmp)
    # Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    if not kekulize:
        Chem.SanitizeMol(new_mol,
                         sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                         | Chem.SanitizeFlags.SANITIZE_PROPERTIES
                         | Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
                         catchErrors=True)
    else:
        Chem.SanitizeMol(new_mol,
                         sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE
                         | Chem.SanitizeFlags.SANITIZE_PROPERTIES
                         | Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
                         catchErrors=True)

    AssignStereochemistry(new_mol,
                          cleanIt=False,
                          force=True,
                          flagPossibleStereoCenters=True)

    new_smiles = Chem.MolToSmiles(new_mol,
                                  isomericSmiles=isomeric,
                                  canonical=canonical)
    return new_smiles


def canonicalize_molecule_smiles(smiles,
                                 return_none_for_error=True,
                                 skip_mol=False,
                                 sort_things=True,
                                 isomeric=True,
                                 kekulization=True,
                                 allow_empty_part=False):
    things = smiles.split('.')
    if skip_mol:
        new_things = things
    else:
        new_things = []
        for thing in things:
            try:
                if thing == '' and not allow_empty_part:
                    raise ValueError('SMILES contains empty part.')

                mol = Chem.MolFromSmiles(thing)
                # print(f"smiles = {thing} mol = {mol}")
                if mol is None:
                    return thing
                assert mol is not None
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                thing_smiles = Chem.MolToSmiles(mol,
                                                kekuleSmiles=False,
                                                isomericSmiles=isomeric)
                thing_smiles = Chem.MolFromSmiles(thing_smiles)
                thing_smiles = Chem.MolToSmiles(thing_smiles,
                                                kekuleSmiles=False,
                                                isomericSmiles=isomeric)
                thing_smiles = Chem.MolFromSmiles(thing_smiles)
                thing_smiles = Chem.MolToSmiles(thing_smiles,
                                                kekuleSmiles=False,
                                                isomericSmiles=isomeric)
                assert thing_smiles is not None
                can_in = thing_smiles
                can_out = canonicalize(thing_smiles, isomeric=isomeric)
                assert can_out is not None, can_in
                thing_smiles = can_out
                if kekulization:
                    thing_smiles = keku_mid = Chem.MolFromSmiles(thing_smiles)
                    assert keku_mid is not None, \
                        'Before can: %s\nAfter can: %s' % (
                            can_in, can_out)
                    thing_smiles = Chem.MolToSmiles(thing_smiles,
                                                    kekuleSmiles=True,
                                                    isomericSmiles=isomeric)
            except KeyboardInterrupt:
                raise
            except Exception:
                if return_none_for_error:
                    return None
                else:
                    raise
            new_things.append(thing_smiles)
    if sort_things:
        new_things = sorted(new_things)
    new_things = '.'.join(new_things)
    return new_things


def canonicalize_reaction_smiles(smiles,
                                 return_none_for_error=True,
                                 return_segs=False,
                                 skip_mol=False,
                                 sort_things=True,
                                 isomeric=True,
                                 kekulization=True):
    segs = smiles.split('>')
    assert len(segs) == 3
    new_segs = []
    for seg in segs:
        if seg != '':
            new_things = canonicalize_molecule_smiles(
                seg,
                return_none_for_error=return_none_for_error,
                skip_mol=skip_mol,
                sort_things=sort_things,
                isomeric=isomeric,
                kekulization=kekulization)
            if return_none_for_error and new_things is None:
                return None
            new_segs.append(new_things)
        else:
            new_segs.append('')

    if return_segs:
        return tuple(new_segs)

    smiles = '>'.join(new_segs)
    return smiles


def get_molecule_id(smiles, remove_duplicate=True):
    if remove_duplicate:
        assert ';' not in smiles
        all_inchi = set()
        for part in smiles.split('.'):
            inchi = get_molecule_id(part, remove_duplicate=False)
            all_inchi.add(inchi)
        all_inchi = tuple(sorted(all_inchi))
        return all_inchi
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ''
        return Chem.MolToInchi(mol)
