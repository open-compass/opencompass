def extract_cpp_code(model_output: str, model_type: str = 'chat'):
    """Extract Cpp Code."""
    if not isinstance(model_output, str):
        return ''

    outputlines = model_output.split('\n')

    indexlines = [i for i, line in enumerate(outputlines) if '```' in line]
    if len(indexlines) >= 2:

        cpp_starts = [
            i for i, line in enumerate(outputlines)
            if '```cpp' in line.lower()
        ]
        if cpp_starts:
            start_index = cpp_starts[0]

            end_indices = [i for i in indexlines if i > start_index]
            if end_indices:
                end_index = end_indices[0]
                extracted = '\n'.join(outputlines[start_index +
                                                  1:end_index]).strip()
                return extracted

        extracted = '\n'.join(outputlines[indexlines[0] +
                                          1:indexlines[-1]]).strip()
        return extracted

    if any(keyword in model_output for keyword in
           ['def ', 'class ', 'int main', '#include', 'using namespace']):
        return model_output.strip()

    return ''


def extract_cpp_code_with_debug(model_output: str, model_type: str = 'chat'):

    if not isinstance(model_output, str):
        return ''

    outputlines = model_output.split('\n')

    # Find Code Block
    indexlines = [i for i, line in enumerate(outputlines) if '```' in line]

    if len(indexlines) >= 2:
        cpp_starts = [
            i for i, line in enumerate(outputlines) if '```c' in line.lower()
        ]

        if cpp_starts:
            start_index = cpp_starts[0]
            end_indices = [i for i in indexlines if i > start_index]
            if end_indices:
                end_index = end_indices[0]
                extracted = '\n'.join(outputlines[start_index +
                                                  1:end_index]).strip()

                return extracted

        extracted = '\n'.join(outputlines[indexlines[0] +
                                          1:indexlines[-1]]).strip()

        return extracted

    keywords = ['def ', 'class ', 'int main', '#include', 'using namespace']
    found_keywords = [kw for kw in keywords if kw in model_output]

    if found_keywords:
        return model_output.strip()

    print('No code found')
    return ''
