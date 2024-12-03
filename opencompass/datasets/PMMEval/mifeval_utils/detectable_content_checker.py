import re


def number_placeholders_checker(input_string: str, num_placeholders: int,
                                **kwargs):
    placeholders = re.findall(r'\[.*?\]', input_string)
    return len(placeholders) >= num_placeholders


def postscript_checker(input_string: str, postscript_marker: str, **kwargs):
    input_string = input_string.lower()
    postscript_pattern = r'\s*' + postscript_marker.lower() + r'.*$'
    postscript = re.findall(postscript_pattern,
                            input_string,
                            flags=re.MULTILINE)
    return True if postscript else False


detectable_content_checker = {
    'number_placeholders': {
        'function': number_placeholders_checker,
        'required_lang_code': False,
        'num_of_params': 2
    },
    'postscript': {
        'function': postscript_checker,
        'required_lang_code': False,
        'num_of_params': 2
    }
}
