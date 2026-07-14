from mmengine.config import read_base

with read_base():
    # scireasoner
    from opencompass.configs.datasets.SciReasoner.bio_instruction_gen import bio_instruction_datasets, \
        mini_bio_instruction_datasets
    from opencompass.configs.datasets.SciReasoner.composition_material_gen import \
        composition_material_datasets, mini_composition_material_datasets
    from opencompass.configs.datasets.SciReasoner.GUE_gen import GUE_datasets, mini_GUE_datasets
    from opencompass.configs.datasets.SciReasoner.smol_gen import all_datasets as smol_datasets, \
        mini_all_datasets as mini_smol_datasets
    from opencompass.configs.datasets.SciReasoner.retrosynthesis_USPTO_gen import \
        Retrosynthesis_datasets as Retrosynthesis_uspto50k_datasets, \
        mini_Retrosynthesis_datasets as mini_Retrosynthesis_uspto50k_datasets
    from opencompass.configs.datasets.SciReasoner.LLM4Mat_gen import LLM4Mat_datasets, mini_LLM4Mat_datasets
    from opencompass.configs.datasets.SciReasoner.bulk_modulus_material_gen import modulus_material_datasets, \
        mini_modulus_material_datasets
    from opencompass.configs.datasets.SciReasoner.mol_biotext_gen import mol_biotext_datasets, mini_mol_biotext_datasets
    from opencompass.configs.datasets.SciReasoner.mol_molecule_gen import mol_mol_datasets, mini_mol_mol_datasets
    from opencompass.configs.datasets.SciReasoner.mol_protein_gen import mol_protein_datasets, mini_mol_protein_datasets
    from opencompass.configs.datasets.SciReasoner.opi_gen import all_datasets as opi_datasets, \
        mini_all_datasets as mini_opi_datasets
    from opencompass.configs.datasets.SciReasoner.peer_gen import PEER_datasets, mini_PEER_datasets
    from opencompass.configs.datasets.SciReasoner.unconditional_material_gen import uncond_material_datasets, \
        mini_uncond_material_datasets
    from opencompass.configs.datasets.SciReasoner.unconditional_RNA_gen import uncond_RNA_datasets, \
        mini_uncond_RNA_datasets
    from opencompass.configs.datasets.SciReasoner.UPG import \
        UPG_datasets as uncond_protein_datasets, mini_UPG_datasets as mini_uncond_protein_datasets
    from opencompass.configs.datasets.SciReasoner.UMG import UMG_Datasets, mini_UMG_Datasets

# full eval set
scireasoner_full_datasets = bio_instruction_datasets + composition_material_datasets + GUE_datasets + smol_datasets + \
    Retrosynthesis_uspto50k_datasets + LLM4Mat_datasets + modulus_material_datasets + \
    mol_biotext_datasets + mol_mol_datasets + mol_protein_datasets + opi_datasets + PEER_datasets + \
    uncond_material_datasets + uncond_RNA_datasets + uncond_protein_datasets + UMG_Datasets

# mini eval set
scireasoner_mini_datasets = mini_bio_instruction_datasets + mini_composition_material_datasets + mini_GUE_datasets + mini_smol_datasets + \
    mini_Retrosynthesis_uspto50k_datasets + mini_LLM4Mat_datasets + mini_modulus_material_datasets + \
    mini_mol_biotext_datasets + mini_mol_mol_datasets + mini_mol_protein_datasets + mini_opi_datasets + mini_PEER_datasets + \
    mini_uncond_material_datasets + mini_uncond_RNA_datasets + mini_uncond_protein_datasets + mini_UMG_Datasets
