import os

charm_tasks = [
    'Chinese_Anachronisms_Judgment',
    'Chinese_Movie_and_Music_Recommendation',
    'Chinese_Natural_Language_Inference',
    'Chinese_Reading_Comprehension',
    'Chinese_Sequence_Understanding',
    'Chinese_Sport_Understanding',
    'Chinese_Time_Understanding',
    'Global_Anachronisms_Judgment',
    'Global_Movie_and_Music_Recommendation',
    'Global_Natural_Language_Inference',
    'Global_Reading_Comprehension',
    'Global_Sequence_Understanding',
    'Global_Sport_Understanding',
    'Global_Time_Understanding',
]

XLT_template = 'Follow the given examples and answer the question.\n{_hint}\n\n I want you to act as an commonsense reasoning expert for Chinese. \n Request: {{input}}\n'
Translate_EN_template = 'Follow the given examples and answer the question.\n{_hint}\n\nQ: {{input}}\nA: '
Other_template = '请按照给定的例子回答问题。\n{_hint}\n\nQ：{{input}}\nA：'

data_dir = 'data/CHARM'
dataset_path_ZH = f'{data_dir}/reasoning'
dataset_path_TransEn = f'{data_dir}/reasoning_Translate-EN'
fewshot_example_path_ZH = os.path.join(os.path.dirname(__file__), 'few-shot-examples')
fewshot_example_path_TransEn = os.path.join(os.path.dirname(__file__), 'few-shot-examples_Translate-EN')

settings = [
    ('Direct', '', dataset_path_ZH, fewshot_example_path_ZH, Other_template),
    ('ZH-CoT', '让我们一步一步来思考。', dataset_path_ZH, fewshot_example_path_ZH, Other_template),
    ('EN-CoT', "Let's think step by step.", dataset_path_ZH, fewshot_example_path_ZH, Other_template),
    ('XLT', """You should retell the request in English.\nYou should do the answer step by step to choose the right answer.\nYou should step-by-step answer the request.\nYou should tell me the answer in this format 'So the answer is'.""", dataset_path_ZH, fewshot_example_path_ZH, XLT_template),
    ('Translate-EN', "Let's think step by step.", dataset_path_TransEn, fewshot_example_path_TransEn, Translate_EN_template),
]
