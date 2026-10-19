import json
import re


def removeprefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix):]
    else:
        return s


def removesuffix(s, suffix):
    if s.endswith(suffix):
        return s[:-len(suffix)]
    else:
        return s


constrained_response = {
    'ar': ['إجابتي هي نعم.', 'إجابتي هي لا.', 'إجابتي هي ربما.'],
    'es':
    ['Mi respuesta es sí.', 'Mi respuesta es no.', 'Mi respuesta es tal vez.'],
    'fr': [
        'Ma réponse est oui.', 'Ma réponse est non.',
        'Ma réponse est peut-être.'
    ],
    'ja': ['私の答えははいです。', '私の答えはいいえです。', '私の答えはたぶんです。'],
    'ko': ['제 대답은 예입니다.', '제 대답은 아닙니다.', '제 대답은 아마도입니다.'],
    'pt': [
        'Minha resposta é sim.', 'Minha resposta é não.',
        'Minha resposta é talvez.'
    ],
    'th': ['คำตอบของฉันคือใช่', 'คำตอบของฉันคือไม่', 'คำตอบของฉันคืออาจจะ'],
    'vi': [
        'Câu trả lời của tôi là có.', 'Câu trả lời của tôi là không.',
        'Câu trả lời của tôi là có thể.'
    ],
    'en': ['My answer is yes.', 'My answer is no.', 'My answer is maybe.'],
    'zh': ['我的答案是是。', '我的答案是否。', '我的答案是不确定。']
}


def constrained_response_checker(input_string: str, lang_code: str, **kwargs):
    allowable_responses = constrained_response[lang_code]
    return any(response in input_string for response in allowable_responses)


def number_bullet_lists_checker(input_string: str, num_bullets: int, **kwargs):
    bullet_lists = re.findall(r'^\s*\*[^\*].*$',
                              input_string,
                              flags=re.MULTILINE)
    bullet_lists_2 = re.findall(r'^\s*-.*$', input_string, flags=re.MULTILINE)
    num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
    return num_bullet_lists == num_bullets


def number_highlighted_sections_checker(input_string: str, num_highlights: int,
                                        **kwargs):
    temp_num_highlights = 0
    highlights = re.findall(r'\*[^\n\*]*\*', input_string)
    double_highlights = re.findall(r'\*\*[^\n\*]*\*\*', input_string)
    for highlight in highlights:
        if highlight.strip('*').strip():
            temp_num_highlights += 1
    for highlight in double_highlights:
        if removesuffix(removeprefix(highlight, '**'), '**').strip():
            temp_num_highlights += 1

    return temp_num_highlights >= num_highlights


def title_checker(input_string: str, **kwargs):
    pattern = r'<<[^\n]+>>'
    re_pattern = re.compile(pattern)
    titles = re.findall(re_pattern, input_string)

    for title in titles:
        if title.lstrip('<').rstrip('>').strip():
            return True
    return False


def json_format_checker(input_string: str, **kwargs):
    value = (removesuffix(
        removeprefix(
            removeprefix(
                removeprefix(removeprefix(input_string.strip(), '```json'),
                             '```Json'), '```JSON'), '```'), '```').strip())
    try:
        json.loads(value)
    except ValueError as e:  # noqa F841
        return False
    return True


detectable_format_checker = {
    'constrained_response': {
        'function': constrained_response_checker,
        'required_lang_code': True,
        'num_of_params': 2
    },
    'json_format': {
        'function': json_format_checker,
        'required_lang_code': False,
        'num_of_params': 1
    },
    'number_bullet_lists': {
        'function': number_bullet_lists_checker,
        'required_lang_code': False,
        'num_of_parmas': 2
    },
    'number_highlighted_sections': {
        'function': number_highlighted_sections_checker,
        'required_lang_code': False,
        'num_of_params': 2
    },
    'title': {
        'function': title_checker,
        'required_lang_code': False,
        'num_of_params': 1
    }
}
