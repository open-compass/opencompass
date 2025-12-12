TASKS = (
    'forward_synthesis',
    'retrosynthesis',
    'molecule_captioning',
    'molecule_generation',
    'name_conversion-i2f',
    'name_conversion-i2s',
    'name_conversion-s2f',
    'name_conversion-s2i',
    'property_prediction-esol',
    'property_prediction-lipo',
    'property_prediction-bbbp',
    'property_prediction-clintox',
    'property_prediction-hiv',
    'property_prediction-sider',
)

DEFAULT_MAX_INPUT_TOKENS = 512
DEFAULT_MAX_NEW_TOKENS = 1024

TASKS_GENERATION_SETTINGS = {
    'forward_synthesis': {
        'generation_kargs': {
            'num_return_sequences': 5,
            'num_beams': 8
        },
    },
    'retrosynthesis': {
        'max_new_tokens': 960,
        'generation_kargs': {
            'num_return_sequences': 10,
            'num_beams': 13
        },
    },
    'molecule_captioning': {
        'generation_kargs': {
            'num_return_sequences': 1,
            'num_beams': 4
        },
    },
    'molecule_generation': {
        'generation_kargs': {
            'num_return_sequences': 5,
            'num_beams': 8
        },
    },
    'name_conversion-i2f': {
        'max_new_tokens': 20,
        'generation_kargs': {
            'num_return_sequences': 3,
            'num_beams': 6
        },
    },
    'name_conversion-i2s': {
        'generation_kargs': {
            'num_return_sequences': 5,
            'num_beams': 8
        },
    },
    'name_conversion-s2f': {
        'max_new_tokens': 20,
        'generation_kargs': {
            'num_return_sequences': 3,
            'num_beams': 6
        },
    },
    'name_conversion-s2i': {
        'generation_kargs': {
            'num_return_sequences': 5,
            'num_beams': 8
        },
    },
    'property_prediction-esol': {
        'batch_size': 16,
        'max_new_tokens': 20,
        'generation_kargs': {
            'num_return_sequences': 1,
            'num_beams': 4,
        },
    },
    'property_prediction-lipo': {
        'batch_size': 16,
        'max_new_tokens': 20,
        'generation_kargs': {
            'num_return_sequences': 1,
            'num_beams': 4,
        },
    },
    'property_prediction-bbbp': {
        'batch_size': 16,
        'max_new_tokens': 20,
        'generation_kargs': {
            'num_return_sequences': 1,
            'num_beams': 4,
        },
    },
    'property_prediction-clintox': {
        'batch_size': 16,
        'max_new_tokens': 20,
        'generation_kargs': {
            'num_return_sequences': 1,
            'num_beams': 4,
        },
    },
    'property_prediction-hiv': {
        'batch_size': 16,
        'max_new_tokens': 20,
        'generation_kargs': {
            'num_return_sequences': 1,
            'num_beams': 4,
        },
    },
    'property_prediction-sider': {
        'batch_size': 16,
        'max_new_tokens': 20,
        'generation_kargs': {
            'num_return_sequences': 1,
            'num_beams': 4,
        },
    },
}

TASK_TAGS = {
    'forward_synthesis': ('<SMILES>', '</SMILES>'),
    'retrosynthesis': ('<SMILES>', '</SMILES>'),
    'molecule_generation': ('<SMILES>', '</SMILES>'),
    'molecule_captioning': (None, None),
    'name_conversion-i2f': ('<MOLFORMULA>', '</MOLFORMULA>'),
    'name_conversion-i2s': ('<SMILES>', '</SMILES>'),
    'name_conversion-s2f': ('<MOLFORMULA>', '</MOLFORMULA>'),
    'name_conversion-s2i': ('<IUPAC>', '</IUPAC>'),
    'property_prediction-esol': ('<NUMBER>', '</NUMBER>'),
    'property_prediction-lipo': ('<NUMBER>', '</NUMBER>'),
    'property_prediction-bbbp': ('<BOOLEAN>', '</BOOLEAN>'),
    'property_prediction-clintox': ('<BOOLEAN>', '</BOOLEAN>'),
    'property_prediction-hiv': ('<BOOLEAN>', '</BOOLEAN>'),
    'property_prediction-sider': ('<BOOLEAN>', '</BOOLEAN>'),
}

# These tasks output SMILES, where there may be semicolons
# that separate different parts. To facilitate evaluation,
# each semicolon is replaced by a dot.
TASKS_WITH_SEMICOLON_REPLACE = (
    'forward_synthesis',
    'retrosynthesis',
    'molecule_generation',
    'name_conversion-i2s',
)

# For these tasks, one input might have multiple gold answers,
# so the gold answer should be directly obtained from the dataset
# instead of directly using the gold domain of each sample.
TASKS_WITH_READING_GOLD_FROM_DATASET = ('forward_synthesis', 'retrosynthesis',
                                        'molecule_generation',
                                        'molecule_captioning',
                                        'name_conversion-i2f',
                                        'name_conversion-i2s',
                                        'name_conversion-s2f',
                                        'name_conversion-s2i')

BASE_MODELS = {
    'osunlp/LlaSMol-Mistral-7B': 'mistralai/Mistral-7B-v0.1',
    'osunlp/LlaSMol-Galactica-6.7B': 'facebook/galactica-6.7b',
    'osunlp/LlaSMol-Llama2-7B': 'meta-llama/Llama-2-7b-hf',
    'osunlp/LlaSMol-CodeLlama-7B': 'codellama/CodeLlama-7b-hf',
}
