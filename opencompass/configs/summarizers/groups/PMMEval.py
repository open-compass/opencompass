NATURAL_LANGUAGE_FULLNAMES = ['English', 'Chinese', 'Arabic', 'Spanish', 'French', 'Japanese', 'Korean', 'Portuguese', 'Thai', 'Vietnamese']
NATURAL_LANGUAGE_FULLNAMES_FLORES = ['Chinese', 'Arabic', 'Spanish', 'French', 'Japanese', 'Korean', 'Portuguese', 'Thai', 'Vietnamese']
NATURAL_LANGUAGE_CODES = ['en', 'zh', 'ar', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi']
NATURAL_LANGUAGE_CODES_MMMLU = ['EN-US', 'ZH-CN', 'AR-XY', 'ES-LA', 'FR-FR', 'JA-JP', 'KO-KR', 'PT-BR', 'TH-TL', 'VI-VT']

PMMEval_summary_groups = [
    {
        'name': 'flores',
        'subsets': [f'flores-{lang_fullname}' for lang_fullname in NATURAL_LANGUAGE_FULLNAMES_FLORES]
    },
    {
        'name': 'humanevalxl',
        'subsets': [f'humanevalxl-python-{lang_fullname}' for lang_fullname in NATURAL_LANGUAGE_FULLNAMES] + \
            [f'humanevalxl-java-{lang_fullname}' for lang_fullname in NATURAL_LANGUAGE_FULLNAMES] + \
            [f'humanevalxl-javascript-{lang_fullname}' for lang_fullname in NATURAL_LANGUAGE_FULLNAMES]
    },
    {
        'name': 'mgsm',
        'subsets': [f'mgsm-{lang_code}' for lang_code in NATURAL_LANGUAGE_CODES]
    },
    {
        'name': 'mhellaswag',
        'subsets': [f'mhellaswag-{lang_code}' for lang_code in NATURAL_LANGUAGE_CODES]
    },
    {
        'name': 'mifeval',
        'subsets': [f'mifeval-{lang_code}' for lang_code in NATURAL_LANGUAGE_CODES]
    },
    {
        'name': 'mlogiqa',
        'subsets': [f'mlogiqa-{lang_code}' for lang_code in NATURAL_LANGUAGE_CODES]
    },
    {
        'name': 'mmmlu',
        'subsets': [f'mmmlu-{lang_code}' for lang_code in NATURAL_LANGUAGE_CODES_MMMLU]
    },
    {
        'name': 'xnli',
        'subsets': [f'xnli-{lang_code}' for lang_code in NATURAL_LANGUAGE_CODES]
    }
]
