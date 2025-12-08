categories = [
        'DNA-cpd',
        'DNA-emp',
        'DNA-pd',
        'DNA-tf-h',
        'DNA-tf-m',
        'Multi_sequence-antibody_antigen',
        'Multi_sequence-promoter_enhancer_interaction',
        'Multi_sequence-rna_protein_interaction',
        'DNA-enhancer_activity',
        'RNA-CRISPROnTarget',
        'Protein-Fluorescence',
        'Protein-Stability',
        'Protein-Thermostability',
        'RNA-Isoform',
        'RNA-MeanRibosomeLoading',
        'RNA-ProgrammableRNASwitches',
        'RNA-Modification',
        'Protein-Solubility',
        'RNA-NoncodingRNAFamily',
        'Protein-FunctionEC',
        'Multi_sequence-sirnaEfficiency',
    ]

biodata_summary_groups = [
    {'name': 'bio_data', 'subsets': [c + '-sample_1k' for c in categories]},
]