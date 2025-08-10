# Copyright LiveCodeBench @ 2024,

import re


def extract_code_generation(model_output: str, model_type: str = 'chat'):
    # modified from
    outputlines = model_output.split('\n')
    # TODO: handle codellama

    if model_type == 'base':
        return model_output.strip()
    elif model_type == 'chat':
        indexlines = [i for i, line in enumerate(outputlines) if '```' in line]
    else:
        raise ValueError(f'Invalid mode type: {model_type}')
    if len(indexlines) < 2:
        return ''
    return '\n'.join(outputlines[indexlines[0] + 1:indexlines[1]])


def extract_code_generation_v2(model_output: str, model_type: str = 'chat'):
    # modified from
    outputlines = model_output.split('\n')
    # TODO: handle codellama

    if model_type == 'base':
        return model_output.strip()
    elif model_type == 'chat':
        indexlines = [i for i, line in enumerate(outputlines) if '```' in line]
    else:
        raise ValueError(f'Invalid mode type: {model_type}')

    if len(indexlines) < 2:
        return ''
    elif len(indexlines) > 2:
        # Only Keep the last code block
        indexlines = indexlines[-2:]

    return '\n'.join(outputlines[indexlines[0] + 1:indexlines[1]])


def extract_code_execution(model_output: str, cot: bool = False):
    pattern = r'\[PYTHON\](.*?)\[\/PYTHON\]'
    matches = re.findall(pattern, model_output, re.DOTALL)
    if matches:
        # fetch the last one
        model_output = matches[-1]

    if '[PYTHON]' in model_output:
        model_output
    if cot:
        if '[ANSWER]' in model_output:
            model_output = model_output.split('[ANSWER]')[1].strip()
    if '==' in model_output:
        model_output = model_output.split('==')[1].strip()
    if '[/ANSWER]' in model_output:
        model_output = model_output.split('[/ANSWER]')[0].strip()
    else:
        model_output = model_output.split('\n')[0].strip()
    return model_output.strip()


def extract_test_output_code(model_output: str):
    outputlines = model_output.split('\n')
    # find the last line startwith assert...
    indexlines = [
        i for i, line in enumerate(outputlines) if line.startswith('assert')
    ]
    if indexlines:
        return outputlines[indexlines[-1]]

    # TODO: handle codellama format
    # if lmstyle and lmstyle == LMStyle.CodeLLaMaInstruct:
    #     indexlines = \
    # [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
    # else:

    # first try to extract ```python if not then try ```
    indexlines = [
        i for i, line in enumerate(outputlines)
        if '```python' in line or '```Python' in line
    ]
    if indexlines:
        start_index = indexlines[0]
    else:
        start_index = None
    indexlines = [i for i, line in enumerate(outputlines) if '```' in line]
    if start_index is not None:
        indexlines = [i for i in indexlines if i > start_index]
        indexlines = [start_index] + indexlines

    if len(indexlines) < 2:
        return ''
    return '\n'.join(outputlines[indexlines[0] + 1:indexlines[1]])
