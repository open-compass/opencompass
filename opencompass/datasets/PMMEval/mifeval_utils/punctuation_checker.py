import re

comma_unicode = {
    'ar': re.compile(r'[\u060C]'),
    'es': re.compile(r'[,\uFF0C]'),
    'fr': re.compile(r'[,\u2026]'),
    'ja': re.compile(r'[,\u3001]'),
    'ko': re.compile(r'[,]'),
    'pt': re.compile(r'[,\uFF0C]'),
    'th': re.compile(r'[\u0E25]'),
    'vi': re.compile(r'[,\uFF0C]'),
    'en': re.compile(r'[,]'),
    'zh': re.compile(r'[,，]')
}


def no_comma_checker(input_string: str, lang_code: str, **kwargs):
    if len(comma_unicode[lang_code].findall(input_string)) > 0:
        return False
    else:
        return True


punctuation_checker = {
    'no_comma': {
        'function': no_comma_checker,
        'required_lang_code': True,
        'num_of_params': 2
    }
}
