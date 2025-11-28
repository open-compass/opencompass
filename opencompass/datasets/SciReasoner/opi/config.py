TASKS = (
    'EC_number',
    'Subcellular_localization',
    'Fold_type',
    'Keywords',
    'GO',
    'Function',
    'gSymbol2Tissue',
    'gSymbol2Cancer',
    'gName2Cancer',
)

DEFAULT_MAX_INPUT_TOKENS = 512
DEFAULT_MAX_NEW_TOKENS = 400

TASKS_GENERATION_SETTINGS = {
    'EC_number': {
        'generation_kargs': {
            'num_return_sequences': 1,
            'num_beams': 1,
            'temperature': 0.2,
            'top_k': 50,
            'top_p': 0.75,
            'do_sample': True,
        },
    },
    'Subcellular_localization': {
        'generation_kargs': {
            'num_return_sequences': 1,
            'num_beams': 1,
            'temperature': 0.2,
            'top_k': 50,
            'top_p': 0.75,
            'do_sample': True,
        },
    },
    'Fold_type': {
        'generation_kargs': {
            'num_return_sequences': 1,
            'num_beams': 1,
            'temperature': 0.2,
            'top_k': 50,
            'top_p': 0.75,
            'do_sample': True,
        },
    },
    'Keywords': {
        'generation_kargs': {
            'num_return_sequences': 1,
            'num_beams': 1,
            'temperature': 0.2,
            'top_k': 50,
            'top_p': 0.75,
            'do_sample': True,
        },
    },
    'GO': {
        'generation_kargs': {
            'num_return_sequences': 1,
            'num_beams': 1,
            'temperature': 0.2,
            'top_k': 50,
            'top_p': 0.75,
            'do_sample': True,
        },
    },
    'Function': {
        'generation_kargs': {
            'num_return_sequences': 1,
            'num_beams': 1,
            'temperature': 0.2,
            'top_k': 50,
            'top_p': 0.75,
            'do_sample': True,
        },
    },
    'gSymbol2Tissue': {
        'generation_kargs': {
            'num_return_sequences': 1,
            'num_beams': 1,
            'temperature': 0.2,
            'top_k': 50,
            'top_p': 0.75,
            'do_sample': True,
        },
    },
    'gSymbol2Cancer': {
        'generation_kargs': {
            'num_return_sequences': 1,
            'num_beams': 1,
            'temperature': 0.2,
            'top_k': 50,
            'top_p': 0.75,
            'do_sample': True,
        },
    },
    'gName2Cancer': {
        'generation_kargs': {
            'num_return_sequences': 1,
            'num_beams': 1,
            'temperature': 0.2,
            'top_k': 50,
            'top_p': 0.75,
            'do_sample': True,
        },
    },
}

TASK_TAGS = {
    'EC_number': ('<EC_NUMBER>', '</EC_NUMBER>'),
    'Subcellular_localization': ('<LOCATION>', '</LOCATION>'),
    'Fold_type': ('<FOLD>', '</FOLD>'),
    'Keywords': ('<KEYWORDS>', '</KEYWORDS>'),
    'GO': ('<GO_TERMS>', '</GO_TERMS>'),
    'Function': ('<FUNCTION>', '</FUNCTION>'),
    'gSymbol2Tissue': ('<TISSUE>', '</TISSUE>'),
    'gSymbol2Cancer': ('<CANCER>', '</CANCER>'),
    'gName2Cancer': ('<CANCER>', '</CANCER>'),
}

# These tasks output SMILES, where there may be semicolons
# that separate different parts.
# To facilitate evaluation, each semicolon is replaced by a dot.
TASKS_WITH_SEMICOLON_REPLACE = ('Keywords', 'GO')

# For these tasks, one input might have multiple gold answers,
# so the gold answer should be directly obtained from the dataset
# instead of directly using the gold domain of each sample.
TASKS_WITH_READING_GOLD_FROM_DATASET = TASKS

BASE_MODELS = {
    'osunlp/LlaSMol-Mistral-7B': 'mistralai/Mistral-7B-v0.1',
    'osunlp/LlaSMol-Galactica-6.7B': 'facebook/galactica-6.7b',
    'osunlp/LlaSMol-Llama2-7B': 'meta-llama/Llama-2-7b-hf',
    'osunlp/LlaSMol-CodeLlama-7B': 'codellama/CodeLlama-7b-hf',
}
