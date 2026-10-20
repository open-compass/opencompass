# flake8: noqa: W605
import re

import timeout_decorator


@timeout_decorator.timeout(5)  # 5 seconds timeout
def safe_regex_search(pattern, text, flags=0):
    try:
        return re.search(pattern, text, flags)
    except timeout_decorator.TimeoutError:
        print(f'Regex match timeout: pattern={pattern}, text={text[:100]}...')
        return None
    except Exception as e:
        print(f'Regex match error: {str(e)}')
        return None


def extract_option_labels(text, options='ABCDEFGHIJ'):
    if not isinstance(text, str) or not isinstance(options, str):
        return 'error'

    text = text.rstrip()
    last_line = text.split('\n')[-1]

    option_str = ''.join([chr(65 + i) for i in range(len(options))
                          ]) if options else 'ABCDEFGHIJ'

    patterns = [
        # e.g. "The final answer to this question is: A."
        #      "The best option is $\boxed{B}:"
        #      "The correct answer is (C)."
        f'[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is?:?\s*(?:[\*\$\\{{(\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){{)?)*\s*([{option_str}])(?:\\\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',

        # e.g. "ANSWER: A"
        #      "Answer: $\boxed{B}."
        #      "ANSWER: (C):"
        f'(?i:Answer)[\*\s]*:\s*(?:[\*\$\\{{(\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){{)?)*\s*([{option_str}])(?:\\\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',

        # e.g. "A"
        #      "$\boxed{B}$"
        #      "(C)."
        #      "[D]:"
        f'^[^\w\r\n]*(?:[\*\$\\{{(\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){{)?)*\s*([{option_str}])(?:\\\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
    ]

    for pattern in patterns:
        match = safe_regex_search(pattern, last_line, re.IGNORECASE)
        if match:
            return match.group(1)

    for pattern in patterns:
        match = safe_regex_search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def extract_option_content(text, options_content=None):
    if not isinstance(text, str) or not isinstance(options_content, list):
        return 'error'

    escaped_options_content = [
        re.escape(option_content) for option_content in options_content
    ]
    escaped_options_content_str = '|'.join(escaped_options_content)

    text = text.rstrip()
    last_line = text.split('\n')[-1]

    patterns = [
        f'[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is:?\s*(?:[\*\$\\{{\(\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){{)?)*\s*({escaped_options_content_str})(?:\\\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
        f'(?i:Answer)\s*(?:[\*\$\\{{\(\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){{)?)*\s*({escaped_options_content_str})(?:\\\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
        f'^[^\w\r\n]*(?:[\*\$\\{{\(\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){{)?)*\s*({escaped_options_content_str})(?:\\\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
    ]

    for pattern in patterns:
        match = safe_regex_search(pattern, last_line)
        if match:
            if match.group(1) in escaped_options_content:
                return options_content[escaped_options_content.index(
                    match.group(1))]
            else:
                return match.group(1)

    for pattern in patterns:
        match = safe_regex_search(pattern, text)
        if match:
            if match.group(1) in escaped_options_content:
                return options_content[escaped_options_content.index(
                    match.group(1))]
            else:
                return match.group(1)

    return None
