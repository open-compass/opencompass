ALL_LANGUAGES = ['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh']
LATIN_LANGUAGES = ['de', 'en', 'es', 'fr', 'sw']
NON_LATIN_LANGUAGES = ['bn', 'ja', 'ru', 'te', 'th', 'zh']

mgsm_summary_groups = [
    {'name': 'mgsm_latin', 'subsets': [f'mgsm_{lang}' for lang in LATIN_LANGUAGES]},
    {'name': 'mgsm_non_latin', 'subsets': [f'mgsm_{lang}' for lang in NON_LATIN_LANGUAGES]},
    {'name': 'mgsm', 'subsets': [f'mgsm_{lang}' for lang in ALL_LANGUAGES]},
]
