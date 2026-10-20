from mmengine.config import read_base

with read_base():
    # scireasoner rawprompt configs
    from opencompass.configs.datasets.SciReasoner.bio_instruction_rawprompt_gen import bio_instruction_datasets, \
        mini_bio_instruction_datasets
    from opencompass.configs.datasets.SciReasoner.composition_material_rawprompt_gen import \
        composition_material_datasets, mini_composition_material_datasets
    from opencompass.configs.datasets.SciReasoner.GUE_rawprompt_gen import GUE_datasets, mini_GUE_datasets
    from opencompass.configs.datasets.SciReasoner.smol_rawprompt_gen import all_datasets as smol_datasets, \
        mini_all_datasets as mini_smol_datasets
    from opencompass.configs.datasets.SciReasoner.retrosynthesis_USPTO_rawprompt_gen import \
        Retrosynthesis_datasets as Retrosynthesis_uspto50k_datasets, \
        mini_Retrosynthesis_datasets as mini_Retrosynthesis_uspto50k_datasets
    from opencompass.configs.datasets.SciReasoner.LLM4Mat_rawprompt_gen import LLM4Mat_datasets, mini_LLM4Mat_datasets
    from opencompass.configs.datasets.SciReasoner.bulk_modulus_material_rawprompt_gen import modulus_material_datasets, \
        mini_modulus_material_datasets
    from opencompass.configs.datasets.SciReasoner.mol_biotext_rawprompt_gen import mol_biotext_datasets, mini_mol_biotext_datasets
    from opencompass.configs.datasets.SciReasoner.mol_molecule_rawprompt_gen import mol_mol_datasets, mini_mol_mol_datasets
    from opencompass.configs.datasets.SciReasoner.mol_protein_rawprompt_gen import mol_protein_datasets, mini_mol_protein_datasets
    from opencompass.configs.datasets.SciReasoner.opi_rawprompt_gen import all_datasets as opi_datasets, \
        mini_all_datasets as mini_opi_datasets
    from opencompass.configs.datasets.SciReasoner.peer_rawprompt_gen import PEER_datasets, mini_PEER_datasets
    from opencompass.configs.datasets.SciReasoner.unconditional_material_rawprompt_gen import uncond_material_datasets, \
        mini_uncond_material_datasets
    from opencompass.configs.datasets.SciReasoner.unconditional_RNA_rawprompt_gen import uncond_RNA_datasets, \
        mini_uncond_RNA_datasets
    from opencompass.configs.datasets.SciReasoner.UPG_rawprompt_gen import \
        UPG_datasets as uncond_protein_datasets, mini_UPG_datasets as mini_uncond_protein_datasets
    from opencompass.configs.datasets.SciReasoner.UMG_rawprompt_gen import UMG_Datasets, mini_UMG_Datasets

# full eval set (rawprompt)
scireasoner_full_datasets = bio_instruction_datasets + composition_material_datasets + GUE_datasets + smol_datasets + \
    Retrosynthesis_uspto50k_datasets + LLM4Mat_datasets + modulus_material_datasets + \
    mol_biotext_datasets + mol_mol_datasets + mol_protein_datasets + opi_datasets + PEER_datasets + \
    uncond_material_datasets + uncond_RNA_datasets + uncond_protein_datasets + UMG_Datasets

# mini eval set (rawprompt)
scireasoner_mini_datasets = mini_bio_instruction_datasets + mini_composition_material_datasets + mini_GUE_datasets + mini_smol_datasets + \
    mini_Retrosynthesis_uspto50k_datasets + mini_LLM4Mat_datasets + mini_modulus_material_datasets + \
    mini_mol_biotext_datasets + mini_mol_mol_datasets + mini_mol_protein_datasets + mini_opi_datasets + mini_PEER_datasets + \
    mini_uncond_material_datasets + mini_uncond_RNA_datasets + mini_uncond_protein_datasets + mini_UMG_Datasets
