flores_summary_groups = []

_flores_lang_map = {
    'Indo-European-Germanic': ['afr', 'dan', 'deu', 'isl', 'ltz', 'nld', 'nob', 'swe'],
    'Indo-European-Romance': ['ast', 'cat', 'fra', 'glg', 'oci', 'por', 'ron', 'spa'],
    'Indo-European-Slavic': ['bel', 'bos', 'bul', 'ces', 'hrv', 'mkd', 'pol', 'rus', 'slk', 'slv', 'srp', 'ukr'],
    'Indo-European-Indo-Aryan': ['asm', 'ben', 'guj', 'hin', 'mar', 'npi', 'ory', 'pan', 'snd', 'urd'],
    'Indo-European-Other': ['ckb', 'cym', 'ell', 'fas', 'gle', 'hye', 'ita', 'lav', 'lit', 'pus', 'tgk'],
    'Austronesian': ['ceb', 'ind', 'jav', 'mri', 'msa', 'tgl'],
    'Atlantic-Congo': ['ibo', 'kam', 'kea', 'lin', 'lug', 'nso', 'nya', 'sna', 'swh', 'umb', 'wol', 'xho', 'yor', 'zul'],
    'Afro-Asiatic': ['amh', 'ara', 'ful', 'mlt', 'orm', 'som'],
    'Turkic': ['azj', 'kaz', 'kir', 'tur', 'uzb'],
    'Dravidian': ['kan', 'mal', 'tam', 'tel'],
    'Sino-Tibetan': ['mya', 'zho_simpl', 'zho_trad'],
    'Other': ['est', 'fin', 'hau', 'heb', 'hun', 'jpn', 'kat', 'khm', 'kor', 'lao', 'luo', 'mon', 'tha', 'vie'],
}
for _lang_serie in _flores_lang_map:
    flores_summary_groups.append({
        'name': f'flores_100_{_lang_serie}_English',
        'subsets': [f'flores_100_{lang_name}-eng' for lang_name in _flores_lang_map[_lang_serie]]
    })
    flores_summary_groups.append({
        'name': f'flores_100_English_{_lang_serie}',
        'subsets': [f'flores_100_eng-{lang_name}' for lang_name in _flores_lang_map[_lang_serie]]
    })

flores_summary_groups.append({
    'name': 'flores_100',
    'subsets': [f'flores_100_{lang_name}-eng' for lang_name in sum(_flores_lang_map.values(), [])] + \
               [f'flores_100_eng-{lang_name}' for lang_name in sum(_flores_lang_map.values(), [])]
})
