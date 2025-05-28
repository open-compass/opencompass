# flake8: noqa
import json
import re

from opencompass.utils import get_logger
from opencompass.datasets.generic import _generic_llmjudge_postprocess

def parse_think(respons):
    index = respons.find("</think>")
    if index != -1:
        return respons[index + len("</think>"):]
    else:
        return respons  


def get_final_results(judged_answers,
                      references,
                      origial_responses,
                      metric_name='accuracy'):
    count = 0
    is_correct_count = 0
    is_incorrect_count = 0
    is_not_attempted_count = 0
    attempted_judge_count = 0
    details = []
    for i, j, k in zip(judged_answers, references, origial_responses):
        if i in ['A', 'B']:
            attempted_judge_count += 1
        grade_letter = i
        detail = {
            'pred': k,
            'ref': j,
            'origin_grade_response': i,
            'grade_letter': grade_letter,
            'correct': False,
        }
        count += 1
        if grade_letter == 'A':
            is_correct_count += 1
            detail['correct'] = True
        elif grade_letter == 'B':
            is_incorrect_count += 1
        else:
            is_not_attempted_count += 1
        details.append(detail)

    is_correct = is_correct_count / count
    is_incorrect = is_incorrect_count / count
    is_given_attempted = is_correct + is_incorrect
    accuracy_given_attempted = (is_correct / is_given_attempted
                                if is_given_attempted > 0 else 0)
    attempted_judge_ratio = attempted_judge_count / count

    f1 = (2 * accuracy_given_attempted * is_correct /
          (accuracy_given_attempted + is_correct) if
          (accuracy_given_attempted + is_correct) > 0 else 0)
    result = {
        metric_name: is_correct * 100,
        f'{metric_name}_given_attempted': accuracy_given_attempted * 100,
        'f1_score': f1,
        'attempted_ratio': attempted_judge_ratio * 100,
        'correct_count': is_correct_count,
        'incorrect_count': is_incorrect_count,
        'not_attempted_count': is_not_attempted_count,
        'details': details,
    }
    return result

def get_numerical_final_results(judged_answers,
                      references,
                      origial_responses,
                      metric_name='mae'):
    sum_abs_error = 0.0
    count = 0
    details = []

    for pred, ref, orig in zip(judged_answers, references, origial_responses):
        error = abs(pred - ref)
        print(pred,ref,error,type(pred),type(ref),type(error))

        sum_abs_error += error
        details.append({
            'pred': pred,
            'ref': ref,
            'origin_response': orig,
            'error': error
        })
        count += 1

    mae = sum_abs_error / count if count > 0 else 0

    result = {
        metric_name: mae,
        'details': details
    }

    return result


def _numerical_postprocess(judgement: str):
    # judgement = parse_think(judgement)
    match = re.search(r'[-+]?\d*\.\d+|\d+\.\d*|\d+', judgement)
    numerical_answer = (match.group(0) if match else 0
                    )  # Return 0 if no match
    return float(numerical_answer)

def numerical_llmjudge_postprocess(
    output: dict,
    output_path: str,
) -> dict:
    judged_answers = []
    origial_responses = []
    references = []
    for k, v in output.items():
        origial_responses.append(v['prediction'])
        processed_judge = _numerical_postprocess(v['prediction'])
        if processed_judge is not None:
            judged_answers.append(processed_judge)
            try:
                references.append(v['gold'])

            except KeyError:
                get_logger().warning(
                    f'No gold answer for {k}, use empty string as reference!')
                references.append(0) #looks like when creating the dataset object the False and 0 value will not be assign a gold value, likely does not inflence the LLM judge, here we just restore the 0 value here.
    results = get_numerical_final_results(judged_answers, references, origial_responses)
    # results['details'] = output
    return results


def generic_llmjudge_postprocess(
    output: dict,
    output_path: str,
) -> dict:
    judged_answers = []
    origial_responses = []
    references = []
    for k, v in output.items():
        origial_responses.append(v['prediction'])
        processed_judge = _generic_llmjudge_postprocess(v['prediction'])
        if processed_judge is not None:
            judged_answers.append(processed_judge)
            try:
                references.append(v['gold'])

            except KeyError:
                get_logger().warning(
                    f'No gold answer for {k}, use empty string as reference!')
                references.append('')
    results = get_final_results(judged_answers, references, origial_responses)
    # results['details'] = output
    return results


def contains_elements_and_matches(sentence, chem_elts):
    matching_elements = [element for element in chem_elts if element in sentence]
    return (bool(matching_elements), matching_elements)

def remove_formula(sentence):
    # 首先尝试识别并移除完整的化学式
    # 匹配常见的化学式模式，包括括号、数字和多个元素的组合
    chemical_formula_pattern = r'\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*(?:\([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\)\d*)*\b'
    sentence = re.sub(chemical_formula_pattern, '', sentence)

    # 识别元素并过滤
    chem_elts = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts']
    contains_elements, matching_elements = contains_elements_and_matches(sentence, chem_elts)

    # 过滤掉答案中剩余的化学元素
    if contains_elements and matching_elements:
        for element in matching_elements:
            # 移除化学符号和可能的化合物
            pattern = re.compile(rf'\b\w*{element}\w*\b')
            sentence = re.sub(pattern, '', sentence)
    return sentence

def verify_float(number):
    if number < 0:
        return abs(number)
    if number >= 0 and number <20:
        return number
    else:
        return 0

def parse_float_answer(sentence):
    # Correctly apply remove_formula to the sentence
    sentence = remove_formula(sentence)
    
    # First, look for formatted answer:number (case-insensitive, no spaces)
    processed_string = sentence.lower().replace(" ", "")
    answer_matches = re.findall(r'answer:(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)', processed_string)
    if answer_matches:
        try:
            return verify_float(float(answer_matches[-1]))
        except ValueError:
            pass
    
    # Then find all scientific notation numbers, take the last one
    sci_matches = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', sentence)
    if sci_matches:
        try:
            return verify_float(float(sci_matches[-1]))
        except ValueError:
            pass
    
    # Lastly, find all regular floats, take the last one
    float_matches = re.findall(r'-?\d+(?:\.\d+)?', sentence)
    if float_matches:
        try:
            return verify_float(float(float_matches[-1]))
        except ValueError:
            pass
    
    # If no valid number found, return 0.0
    return 0.0




def parse_true_false_answer(raw_string, option=''):
    # print("yes this is called")
    sentence = remove_formula(raw_string)
    answer_striped = raw_string.lower().replace(" ", "")
    if 'answer:true' in answer_striped:
        return True
    elif 'answer:false' in answer_striped:
        return False
    elif "not" in answer_striped:
        return False
    elif "no" in answer_striped:
        return False
    elif "yes" in answer_striped:
        return True
    elif "itis" in answer_striped:
        return True
    else:
        return True


def parse_has_hasnot_answer(raw_string, option=''):
    sentence = remove_formula(raw_string)
    answer_striped = raw_string.lower().replace(" ", "")
    if 'answer:true' in answer_striped:
        return True
    elif 'answer:false' in answer_striped:
        return False
    elif "doesnot" in answer_striped:
        return False    
    elif "not" in answer_striped:
        return False
    elif "no" in answer_striped:
        return False
    elif "yes" in answer_striped:
        return True
    elif "itis" in answer_striped:
        return True
    elif "has" in answer_striped:
        return True
    else:
        return True
