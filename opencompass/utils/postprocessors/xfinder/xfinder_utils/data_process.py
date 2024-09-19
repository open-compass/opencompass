import ast


class DataProcessor:

    def __init__(self):
        pass

    def read_data(self, data):
        for item in data:
            if isinstance(item['standard_answer_range'],
                          str) and item['key_answer_type'] != 'math':
                try:
                    item['standard_answer_range'] = ast.literal_eval(
                        item['standard_answer_range'])
                except Exception as e:
                    print(f'Error: {e}')
                    print('Please check the form of standard_answer_range')
                    exit(0)

            item['standard_answer_range'] = str(item['standard_answer_range'])
            item['key_answer_type'] = str(item['key_answer_type'])

        return data
