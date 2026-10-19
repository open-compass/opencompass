def repeat_prompt_checker(input_string: str, prompt_to_repeat: str, **kwargs):
    if input_string.strip().lower().startswith(
            prompt_to_repeat.strip().lower()):
        return True
    return False


def two_responses_checker(input_string: str, **kwargs):
    valid_responses = list()
    responses = input_string.split('******')
    for index, response in enumerate(responses):
        if not response.strip():
            if index != 0 and index != len(responses) - 1:
                return False
        else:
            valid_responses.append(response)
    return (len(valid_responses) == 2
            and valid_responses[0].strip() != valid_responses[1].strip())


combination_checker = {
    'repeat_prompt': {
        'function': repeat_prompt_checker,
        'required_lang_code': False,
        'num_of_params': 2
    },
    'two_responses': {
        'function': two_responses_checker,
        'required_lang_code': False,
        'num_of_params': 1
    }
}
