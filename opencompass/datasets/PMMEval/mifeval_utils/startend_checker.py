def end_checker_checker(input_string: str, end_phrase: str, **kwargs):
    if input_string.strip().endswith(end_phrase):
        return True
    else:
        return False


def quotation_checker(input_string: str, lang_code: str, **kwargs):
    input_string = input_string.strip()
    if input_string.startswith('"') and input_string.endswith('"'):
        return True
    elif lang_code in [
            'ar', 'es', 'fr', 'pt', 'ru'
    ] and input_string.startswith('«') and input_string.endswith('»'):
        return True
    elif lang_code in [
            'ar', 'es', 'fr', 'ko', 'pt', 'th', 'vi', 'zh'
    ] and input_string.startswith('“') and input_string.endswith('”'):
        return True
    elif lang_code == 'ja' and input_string.startswith(
            '『') and input_string.endswith('』'):
        return True
    else:
        return False


startend_checker = {
    'end_checker': {
        'function': end_checker_checker,
        'required_lang_code': False,
        'num_of_params': 2
    },
    'quotation': {
        'function': quotation_checker,
        'required_lang_code': True,
        'num_of_params': 2
    }
}
