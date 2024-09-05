# Convert OpenCompass prediction data to XFinder format
import copy
import json
import re

xfinder_template = {
    'math': {
        'model_name':
        '',
        'dataset':
        '',
        'key_answer_type':
        'math',
        'question':
        '',
        'llm_output':
        '',
        'correct_answer':
        '',
        'standard_answer_range':
        'a(n) number / set / vector / matrix / interval / expression / function / equation / inequality'  # noqa
    },
    'alphabet_option': {
        'model_name': '',
        'dataset': '',
        'key_answer_type': 'alphabet_option',
        'question': '',
        'llm_output': '.',
        'correct_answer': '',
        'standard_answer_range': []
    },
    'categorical_label': {
        'model_name': '',
        'dataset': '',
        'key_answer_type': '',
        'question': '',
        'llm_output': '',
        'correct_answer': '',
        'standard_answer_range': []
    },
    'short_text': {
        'model_name': '',
        'dataset': '',
        'key_answer_type': 'short_text',
        'question': '',
        'llm_output': '',
        'correct_answer': '',
        'standard_answer_range': []
    }
}


def parse_options(text: str):
    lines = text.split('\n')
    parsed_options = []
    option_pattern = r'^[A-Z]\)|[A-Z]\.|[A-Z]\)|[A-Z]:|\([A-Z]\)'
    for line in lines:
        line = line.strip()
        match = re.match(option_pattern, line)
        if match:
            option = ''
            # 等于第一个属于选项的字符
            for c in line:
                if c.isalpha():
                    option = c
                    break
            content_start = match.end() + 1
            content = line[content_start:].strip()
            parsed_options.append([option, content])

    return parsed_options


def convert_to_xfinder_format(typ, data, model_name='', dataset_name=''):
    assert typ in xfinder_template.keys(), f'Invalid type {typ}'
    format_data = []
    for item in data:
        template = copy.deepcopy(xfinder_template[typ])
        question = item['origin_prompt'][-1]['prompt']
        llm_output = item['prediction']
        correct_answer = item['reference'] if item['reference'] else item[
            'gold']
        template['correct_answer'] = correct_answer
        template['model_name'] = model_name
        template['dataset'] = dataset_name
        template['question'] = question
        template['llm_output'] = llm_output
        try:
            assert typ in list(xfinder_template.keys())
            if typ == 'alphabet_option':
                options = parse_options(question)
                template['standard_answer_range'] = options
            elif typ == 'short_text':
                template['standard_answer_range'] = item['gold']
            elif typ == 'categorical_label':
                pass
        except Exception as e:
            print(f'Error when parsing question options: {e}, skipping...')
            continue

        format_data.append(template)
    return format_data


if __name__ == '__main__':
    # Test
    example_data = {
        'origin_prompt': [{
            'role':
            'HUMAN',
            'prompt':
            'Alice, Bob, Claire, Dave, and Eve are dancers at a square dance. At the start of a song, they each have a partner: Alice is dancing with Ophelia, Bob is dancing with Jamie, Claire is dancing with Melissa, Dave is dancing with Rodrigo, and Eve is dancing with Patrick.\nThroughout the song, the dancers often trade partners. First, Claire and Bob switch partners. Then, Claire and Eve switch partners. Then, Claire and Bob switch partners. Then, Eve and Dave switch partners. Finally, Claire and Alice switch partners. At the end of the dance, Alice is dancing with\nOptions:\n(A) Ophelia\n(B) Jamie\n(C) Melissa\n(D) Rodrigo\n(E) Patrick'  # noqa
        }],
        'origin_prediction':
        '\n 答案: B) 前者小于后者',
        'prediction':
        'B',
        'reference':
        'A'
    }
    example_data = convert_to_xfinder_format('alphabet_option', [example_data],
                                             'GPT-3', 'OpenAI')
    print(json.dumps(example_data, indent=4, ensure_ascii=False))
