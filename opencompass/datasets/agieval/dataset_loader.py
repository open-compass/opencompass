# flake8: noqa
import ast
import json
import os
from os import environ

import pandas as pd
import tiktoken
from tqdm import tqdm

from .constructions import ChatGPTSchema, ResultsForHumanSchema
from .utils import extract_answer, read_jsonl, save_jsonl

# define the datasets
english_qa_datasets = [
    'lsat-ar', 'lsat-lr', 'lsat-rc', 'logiqa-en', 'sat-math', 'sat-en',
    'aqua-rat', 'sat-en-without-passage', 'gaokao-english'
]
chinese_qa_datasets = [
    'logiqa-zh', 'jec-qa-kd', 'jec-qa-ca', 'gaokao-chinese',
    'gaokao-geography', 'gaokao-history', 'gaokao-biology', 'gaokao-chemistry',
    'gaokao-physics', 'gaokao-mathqa'
]
english_cloze_datasets = ['math']
chinese_cloze_datasets = ['gaokao-mathcloze']

multi_choice_datasets = ['jec-qa-kd', 'jec-qa-ca', 'gaokao-physics']
math_output_datasets = ['gaokao-mathcloze', 'math']


def convert_zero_shot(line, dataset_name):
    try:
        passage = line['passage'] if line['passage'] is not None else ''
        if dataset_name in english_qa_datasets:
            option_string = 'ABCDEFG'
            count = len(line['options'])
            if count == 1:
                count = 5
            return passage + 'Q: '  + line['question'] + ' ' \
                + 'Answer Choices: ' + ' '.join(line['options']) + '\n' + \
                'A: Among A through {}, the answer is'.format(option_string[count - 1])

        elif dataset_name in chinese_qa_datasets:
            option_string = 'ABCDEFG'
            count = len(line['options'])
            if count == 1:
                count = 4
            return passage + '问题：' + line['question'] + ' ' \
                + '选项：' + ' '.join(line['options']) + '\n' + \
                '答案：从A到{}, 我们应选择'.format(option_string[count - 1])

        elif dataset_name in english_cloze_datasets:
            return passage + 'Q: ' + line['question'] + '\n' \
                                              'A: The answer is'

        elif dataset_name in chinese_cloze_datasets:
            return passage + '问题：' + line['question'] + '\n' \
                                                '答案：'
    except NameError:
        print('Dataset not defined.')


prefix = '该问题为单选题，所有选项中必有一个正确答案，且只有一个正确答案。\n'


def convert_zero_shot_CoT_stage1(line, dataset_name):
    try:
        passage = line['passage'] if line['passage'] is not None else ''
        if dataset_name in english_qa_datasets:
            return passage + 'Q: ' + line['question'] + ' ' \
                + 'Answer Choices: ' + ' '.join(line['options']) + '\n' + \
                "Let's think step by step."

        elif dataset_name in chinese_qa_datasets:
            option_string = 'ABCDEFG'
            count = len(line['options'])
            if count == 1:
                count = 4
            return passage + '问题：' + line['question'] + ' ' \
                + '选项：' + ' '.join(line['options']) + '\n' + \
                '从A到{}, 我们应选择什么？让我们逐步思考：'.format(option_string[count - 1])

        elif dataset_name in english_cloze_datasets:
            return passage + 'Q: ' + line['question'] + '\n' \
                                              "A: Let's think step by step."

        elif dataset_name in chinese_cloze_datasets:
            return passage + '问题：' + line['question'] + '\n' \
                                                '答案：让我们逐步思考：'
    except NameError:
        print('Dataset not defined.')


# process few-shot raw_prompts
def combine_prompt(prompt_path,
                   dataset_name,
                   load_explanation=True,
                   chat_mode=False):
    skip_passage = False
    if dataset_name == 'sat-en-without-passage':
        skip_passage = True
        dataset_name = 'sat-en'
    demostrations = []
    # read the prompts by context and explanation
    context_row = [0, 1, 3, 5, 7, 9]
    explanation_row = [0, 2, 4, 6, 8, 10]
    raw_prompts_context = pd.read_csv(prompt_path,
                                      header=0,
                                      skiprows=lambda x: x not in context_row,
                                      keep_default_na=False)
    raw_prompts_explanation = pd.read_csv(
        prompt_path,
        header=0,
        skiprows=lambda x: x not in explanation_row,
        keep_default_na=False).replace(r'\n\n', '\n', regex=True)
    contexts = []
    for line in list(raw_prompts_context[dataset_name]):
        if line:
            # print(line)
            contexts.append(ast.literal_eval(line))
    explanations = [
        exp for exp in raw_prompts_explanation[dataset_name] if exp
    ]

    for idx, (con, exp) in enumerate(zip(contexts, explanations)):
        passage = con['passage'] if con[
            'passage'] is not None and not skip_passage else ''
        question = con['question']
        options = con['options'] if con['options'] is not None else ''
        label = con['label'] if con['label'] is not None else ''
        answer = con[
            'answer'] if 'answer' in con and con['answer'] is not None else ''

        if dataset_name in english_qa_datasets:
            question_input = 'Problem {}.   '.format(idx + 1) + passage + ' ' + question + '\n' \
                              + 'Choose from the following options:    ' + ' '.join(options) + '\n'
            question_output = (('Explanation for Problem {}:   '.format(idx + 1) + exp + '\n') if load_explanation else '') \
                              + 'The answer is therefore {}'.format(label)

        elif dataset_name in chinese_qa_datasets:
            question_input = '问题 {}.   '.format(idx + 1) + passage + ' ' + question + '\n' \
                              + '从以下选项中选择:    ' + ' '.join(options) + '\n'
            question_output = (('问题 {}的解析:   '.format(idx + 1) + exp + '\n') if load_explanation else '') \
                              + '答案是 {}'.format(label)

        elif dataset_name in english_cloze_datasets:
            question_input = 'Problem {}.   '.format(idx + 1) + question + '\n'
            question_output = (('Explanation for Problem {}:   '.format(idx + 1) + exp + '\n') if load_explanation else '') \
                              + 'The answer is therefore {}'.format(answer)

        elif dataset_name in chinese_cloze_datasets:
            question_input = '问题 {}.   '.format(idx + 1) + question + '\n'
            question_output = (('问题 {}的解析:   '.format(idx + 1) + exp + '\n') if load_explanation else '') \
                              + '答案是 {}'.format(answer)
        else:
            raise ValueError(
                f'During loading few-sot examples, found unknown dataset: {dataset_name}'
            )
        if chat_mode:
            demostrations.append((question_input, question_output))
        else:
            demostrations.append(question_input + question_output + '\n')

    return demostrations


enc = None


def _lazy_load_enc():
    global enc
    if enc is None:
        enc = tiktoken.encoding_for_model('gpt-4')


# cut prompt if reach max token length
def concat_prompt(demos,
                  dataset_name,
                  max_tokens,
                  end_of_example='\n',
                  verbose=False):
    _lazy_load_enc()
    demostration_en = 'Here are the answers for the problems in the exam.\n'
    demostration_zh = '以下是考试中各个问题的答案。\n'

    for i in range(len(demos)):
        # print(len(enc.encode(demostration_en)), len(enc.encode(demostration_zh)))
        if dataset_name in english_qa_datasets:
            demostration_en = demostration_en + demos[i] + end_of_example
        elif dataset_name in chinese_qa_datasets:
            demostration_zh = demostration_zh + demos[i] + end_of_example
        elif dataset_name in english_cloze_datasets:
            demostration_en = demostration_en + demos[i] + end_of_example
        elif dataset_name in chinese_cloze_datasets:
            demostration_zh = demostration_zh + demos[i] + end_of_example
        # break if reach max token limit
        if len(enc.encode(demostration_en)) < max_tokens and len(
                enc.encode(demostration_zh)) < max_tokens:
            output = demostration_en if len(demostration_en) > len(
                demostration_zh) else demostration_zh
            prompt_num = i + 1
        else:
            break
    if verbose:
        print('max_tokens set as ', max_tokens, 'actual_tokens is',
              len(enc.encode(output)), 'num_shot is', prompt_num)
    return output, prompt_num


def concat_prompt_chat_mode(demos,
                            dataset_name,
                            max_tokens,
                            end_of_example='\n',
                            verbose=False):
    _lazy_load_enc()
    answers = []
    sentences = ''
    for i in range(len(demos)):
        answers += [
            {
                'role': 'user',
                'content': demos[i][0]
            },
            {
                'role': 'assistant',
                'content': demos[i][1]
            },
        ]
        sentences += json.dumps(answers[-1])
        # break if reach max token limit
        if len(enc.encode(sentences)) > max_tokens:
            answers.pop()
            answers.pop()
            break
    if verbose:
        print('max_tokens set as ', max_tokens, 'actual_tokens is',
              len(enc.encode(sentences)), 'num_shot is',
              len(answers) // 2)
    return answers, len(answers) // 2


def convert_few_shot(line, dataset_name, demo, n_shot, chat_mode=False):
    passage = line['passage'] if line['passage'] is not None else ''
    question = line['question']
    options = line['options'] if line['options'] is not None else ''

    if dataset_name in english_qa_datasets:
        question_input = 'Problem {}.   '.format(n_shot + 1) + passage + ' ' + question + '\n' \
            + 'Choose from the following options:    ' + ' '.join(options) + '\n'
        # + "Explanation for Problem {}:   ".format(n_shot + 1)

    if dataset_name in chinese_qa_datasets:
        question_input = '问题 {}.   '.format(n_shot + 1) + passage + ' ' + question + '\n' \
            + '从以下选项中选择:    ' + ' '.join(options) + '\n'
        # + "问题 {}的解析:   ".format(n_shot + 1)

    if dataset_name in english_cloze_datasets:
        question_input = 'Problem {}.   '.format(n_shot + 1) + question + '\n'
        # + "Explanation for Problem {}:   ".format(n_shot + 1)

    if dataset_name in chinese_cloze_datasets:
        question_input = '问题 {}.   '.format(n_shot + 1) + question + '\n'
        # + "问题 {}的解析:   ".format(n_shot + 1)
    if chat_mode:
        return demo + [
            {
                'role': 'user',
                'content': question_input
            },
        ]
    else:
        return demo + question_input


def load_dataset(dataset_name,
                 setting_name,
                 parent_path,
                 prompt_path=None,
                 max_tokens=None,
                 end_of_example='\n',
                 chat_mode=False,
                 verbose=False):

    if environ.get('DATASET_SOURCE') == 'ModelScope':
        from modelscope import MsDataset
        loaded_jsonl = MsDataset.load(parent_path,
                                      subset_name=dataset_name,
                                      split='test')
    else:
        test_path = os.path.join(parent_path, dataset_name + '.jsonl')
        loaded_jsonl = read_jsonl(test_path)
    processed = []
    if setting_name == 'few-shot-CoT' or setting_name == 'few-shot':
        # process demo once if it is few-shot-CoT
        processed_demos = combine_prompt(
            prompt_path,
            dataset_name,
            load_explanation=setting_name == 'few-shot-CoT',
            chat_mode=chat_mode)
        if chat_mode:
            chosen_prompt, n_shot = concat_prompt_chat_mode(processed_demos,
                                                            dataset_name,
                                                            max_tokens,
                                                            end_of_example,
                                                            verbose=verbose)
        else:
            chosen_prompt, n_shot = concat_prompt(processed_demos,
                                                  dataset_name,
                                                  max_tokens,
                                                  end_of_example,
                                                  verbose=verbose)
    if verbose:
        loaded_jsonl = tqdm(loaded_jsonl)
    for meta_idx, line in enumerate(loaded_jsonl):
        if setting_name == 'zero-shot':
            ctxt = convert_zero_shot(line, dataset_name)
        elif setting_name == 'zero-shot-CoT':
            ctxt = convert_zero_shot_CoT_stage1(line, dataset_name)
        elif setting_name == 'few-shot-CoT' or setting_name == 'few-shot':
            ctxt = convert_few_shot(line, dataset_name, chosen_prompt, n_shot,
                                    chat_mode)
        try:
            new_instance = ChatGPTSchema(context=ctxt, metadata=meta_idx)
            processed.append(new_instance.to_dict())
        except NameError:
            print('Dataset not defined.')
    return processed


def generate_second_stage_input(dataset_name,
                                input_list,
                                output_list,
                                with_format_prompt=False):
    try:
        english_format_prompt = 'Based on the previous results, your task is to extract the final answer and provide the output enclosed in brackets【】, such as 【0】 or 【A】.'
        chinese_format_prompt = '根据以上内容，你的任务是把最终的答案提取出来并填在【】中，例如【0】或者【A】。'
        if dataset_name in english_qa_datasets:
            prompt_suffix = 'Therefore, among A through E, the answer is'
            if with_format_prompt:
                prompt_suffix = english_format_prompt + prompt_suffix
        elif dataset_name in chinese_qa_datasets:
            prompt_suffix = '因此，从A到D, 我们应选择'
            if with_format_prompt:
                prompt_suffix = chinese_format_prompt + prompt_suffix
        elif dataset_name in english_cloze_datasets:
            prompt_suffix = 'Therefore, the answer is'
            if with_format_prompt:
                prompt_suffix = english_format_prompt + prompt_suffix
        elif dataset_name in chinese_cloze_datasets:
            prompt_suffix = '因此，答案是'
            if with_format_prompt:
                prompt_suffix = chinese_format_prompt + prompt_suffix
    except NameError:
        print('Dataset not defined.')
    processed = []
    for i in range(len(input_list)):
        ctxt = '{0}\n{1}\n{2}'.format(input_list[i]['context'],
                                      extract_answer(output_list[i]),
                                      prompt_suffix)
        new_instance = ChatGPTSchema(context=ctxt,
                                     metadata=input_list[i]['metadata'])
        processed.append(new_instance.to_dict())
    return processed


def load_dataset_as_result_schema(dataset_name, parent_path):

    if environ.get('DATASET_SOURCE') == 'ModelScope':
        from modelscope import MsDataset
        loaded_jsonl = MsDataset.load(parent_path,
                                      subset_name=dataset_name,
                                      split='test')
    else:
        test_path = os.path.join(parent_path, dataset_name + '.jsonl')
        loaded_jsonl = read_jsonl(test_path)

    processed = []
    for i, line in enumerate(loaded_jsonl):
        problem_input = convert_zero_shot(line, dataset_name)
        processed.append(
            ResultsForHumanSchema(
                index=i,
                problem_input=problem_input,
                label=line['label'] if line['label'] else line['answer'],
            ))
    return processed


if __name__ == '__main__':

    # set variables
    parent_dir = '../../data/V1_1/'
    raw_prompt_path = '../data/few_shot_prompts.csv'

    # set dataset name to process
    setting_name = 'few-shot-CoT'  # setting_name can be chosen from ["zero-shot", "zero-shot-CoT", "few-shot-CoT"]
    data_name = 'jec-qa-kd'
    save_dir = '../../experiment_input/{}/'.format(setting_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    processed_data = load_dataset(data_name,
                                  setting_name,
                                  parent_dir,
                                  prompt_path=raw_prompt_path,
                                  max_tokens=2048)
    save_jsonl(processed_data,
               os.path.join(save_dir, '{}.jsonl'.format(data_name)))
