import re
from string import Formatter


def format_string(template: str, input_data: dict) -> str:
    """Return string with input content according input format template.

    Args:
        template (str): Format string with keyword-only argument. For
            example '{who} like {what}'
        input_data (dict): Input data to fill in the input template.

    Returns:
        str: Return string.
    """

    return template.format(**input_data)


def parse_string(template: str, input_string: str, allow_newline: bool=False) -> dict:
    """Return a dictionary whose keys are from input template and value is
    responding content from input_string.

    Args:
        template (str): Format template with keyword-only argument. For
            example '{who} like {what}'
        input_string (str): Input string will be parsed.
        allow_newline (boolen): Whether allow '\n' in {} during RE match, default to False.

    Returns:
        dict: Parsed data from input string according to format string. If
            input string doesn't match template, It will return None.

    Examples:
        >>> template = '{who} like {what}'
        >>> input_string = 'monkey like banana'
        >>> data = parse_string(template, input_string)
        >>> data
        >>> {'who': 'monkey', 'what': 'banana'}
        >>> input_string = 'monkey likes banana'
        >>> data = parse_string(template, input_string)
        >>> data
        >>> None
        >>> template = '{what} like {what}'
        >>> input_string = 'monkey like banana'
        >>> data = parse_string(template, input_string)
        >>> data
        >>> {'what': ['monkey', 'banana']}
    """

    formatter = Formatter()
    context = []
    keys = []
    for v in formatter.parse(template):
        # v is (literal_text, field_name, format_spec, conversion)
        if v[1] is not None:
            keys.append(v[1])
        context.append(v[0])
    pattern = template
    for k in keys:
        pattern = pattern.replace('{' + f'{k}' + '}', '(.*)')
    # pattern = re.compile(rf'{pattern}')
    values = re.findall(pattern, input_string, re.S if allow_newline else 0)
    if len(values) < 1:
        return None
    data = dict()
    for k, v in zip(keys, values[0]):
        if k in data:
            tmp = data[k]
            if isinstance(tmp, list):
                data[k].append(v)
            else:
                data[k] = [tmp, v]
        else:
            data[k] = v
    return data
