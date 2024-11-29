def forbidden_words_checker(input_string: str, forbidden_words: list,
                            **kwargs):
    return not any(word in input_string for word in forbidden_words)


keywords_checker = {
    'forbidden_words': {
        'function': forbidden_words_checker,
        'required_lang_code': False,
        'num_of_params': 2
    },
}
