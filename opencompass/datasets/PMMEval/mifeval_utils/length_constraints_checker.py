import re


def nth_paragraph_first_word_checker(input_string: str, num_paragraphs: int,
                                     nth_paragraph: int, first_word: str,
                                     lang_code: str, **kwargs):
    paragraphs = re.split(r'\n\n', input_string)
    paragraphs = list(paragraph.strip() for paragraph in paragraphs
                      if paragraph.strip() != '')

    if len(paragraphs) < num_paragraphs:
        return False

    if len(paragraphs) < nth_paragraph:
        return False

    paragraph = paragraphs[nth_paragraph - 1].strip()

    first_word = ''

    if paragraph.lower().startswith(first_word.lower()):
        return True
    else:
        return False


def number_paragraphs_checker(input_string: str, num_paragraphs: int,
                              **kwargs):
    paragraphs = re.split(r'\s?\*\*\*\s?', input_string)
    paragraphs = list(paragraph.strip() for paragraph in paragraphs
                      if paragraph.strip() != '')
    return len(paragraphs) == num_paragraphs


def number_sentences_checker(input_string: str, relation: str,
                             num_sentences: int, lang_code: str, **kwargs):
    sentences = list(x.strip() for x in input_string.strip().split('\n'))
    sentences = list(x for x in sentences if x != '')

    if relation == 'less than':
        if len(sentences) <= num_sentences:
            return True
        else:
            return False
    elif relation == 'at least':
        if len(sentences) >= num_sentences:
            return True
        else:
            return False


def number_words_checker(input_string: str, relation: str, num_words: int,
                         lang_code: str, **kwargs):
    if lang_code in ['en', 'es', 'fr', 'in', 'pt', 'ru', 'vi']:
        words = input_string.split()
        words = list(x for x in words if x != '')
    else:
        words = ''.join(input_string.split())

    if relation == 'less than':
        if len(words) <= num_words:
            return True
        else:
            return False
    elif relation == 'at least':
        if len(words) >= num_words:
            return True
        else:
            return False


length_constraints_checker = {
    'nth_paragraph_first_word': {
        'function': nth_paragraph_first_word_checker,
        'required_lang_code': True,
        'num_of_params': 5
    },
    'number_paragraphs': {
        'function': number_paragraphs_checker,
        'required_lang_code': False,
        'num_of_params': 2
    },
    'number_sentences': {
        'function': number_sentences_checker,
        'required_lang_code': True,
        'num_of_params': 3
    },
    'number_words': {
        'function': number_words_checker,
        'required_lang_code': True,
        'num_of_params': 4
    }
}
