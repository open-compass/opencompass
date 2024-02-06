import ast
import json
def format_load(raw_data: str, start_character: str = '', end_character: str = ''):
    """Format the raw data into the format that can be evaluated.

    Args:
        raw_data (str): The raw data.
        start_character (str, optional): The start character. Defaults to '', if using it, the string will be sliced from the first start_character.
        end_character (str, optional): The end character. Defaults to '', if using it, the string will be sliced to the last end_character.

    Returns:
        str: The formatted data.
    """
    if type(raw_data) != str:
        # the data has been evaluated
        return raw_data
    if "```json" in raw_data:
        raw_data = raw_data[raw_data.find("```json") + len("```json"):]
        raw_data = raw_data.strip("`")
    if start_character != '':
        raw_data = raw_data[raw_data.find(start_character):]
    if end_character != '':
        raw_data = raw_data[:raw_data.rfind(end_character) + len(end_character)]
    successful_parse = False
    try:
        data = ast.literal_eval(raw_data)
        successful_parse = True
    except Exception as e:
        pass
    try:
        if not successful_parse:
            data = json.loads(raw_data)
        successful_parse = True
    except Exception as e:
        pass
    try:
        if not successful_parse:
            data = json.loads(raw_data.replace("\'", "\""))
        successful_parse = True
    except Exception as e:
        pass
    if not successful_parse:
        raise Exception("Cannot parse raw data")
    return data
