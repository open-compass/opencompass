import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (accuracy_score, matthews_corrcoef,
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, roc_auc_score)
from tqdm import tqdm
from transformers import pipeline

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator

current_working_directory = os.getcwd()
path_bioinstruction = os.path.join(current_working_directory, 'opencompass',
                                   'datasets', 'bioinstruction')

print(torch.cuda.is_available())

classifier = pipeline('zero-shot-classification',
                      model='facebook/bart-large-mnli',
                      device=0)


# @LOAD_DATASET.register_module()
class Bioinstruction_Dataset(BaseDataset):

    @staticmethod
    def load(path, train_path, test_path, mini_set=False, hf_hub=False):
        if (hf_hub is True):
            # load from huggingface hub
            train_data = []
            repo_id = test_path.split('/')[0] + '/' + test_path.split('/')[1]
            train_path = train_path.split(repo_id + '/')[1]
            test_path = test_path.split(repo_id + '/')[1]
            train_path = hf_hub_download(repo_id,
                                         train_path,
                                         repo_type='dataset')
            test_path = hf_hub_download(repo_id,
                                        test_path,
                                        repo_type='dataset')

        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            train_data = train_data[:5]
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        selected_train_data = [{
            'input': record['input'],
            'output': record['output']
        } for record in train_data]
        selected_test_data = [{
            'input': record['input'],
            'output': record['output']
        } for record in test_data]
        # dataset=Dataset.from_list(selected_train_data)
        if mini_set and len(selected_test_data) > 150:
            import random
            random.seed(1024)
            selected_test_data = random.sample(selected_test_data, 150)
            random.seed()

        dataset = DatasetDict({
            'train': Dataset.from_list(selected_train_data),
            'test': Dataset.from_list(selected_test_data)
        })
        return dataset


def extract_answer_part(outputs, left_tag, right_tag, mode='tag'):
    assert mode in ('tag', 'direct')

    assert isinstance(outputs, list)
    answers = []
    for text in outputs:
        if mode == 'direct' or (left_tag is None and right_tag is None):
            text = text.replace('<unk>', '').replace('</s>', '').strip()
            answers.append(text.strip())
            continue

        left_tag_pos = text.find(left_tag)
        if left_tag_pos == -1:
            answers.append('')
            continue
        right_tag_pos = text.find(right_tag)
        if right_tag_pos == -1:
            answers.append('')
            continue
        text = text[left_tag_pos + len(left_tag):right_tag_pos].strip()
        answers.append(text)
    return answers


def extract_numeric_values(text):
    text = text.replace("5'", "five'")
    text = text.replace("3'", 'three')

    matches = re.findall(r'(?<![a-zA-Z])[-‑]?\d+\.?\d*', str(text))
    # matches = re.findall(r"(?<![a-zA-Z])[-‑]?\d+\.?\d*", str(text))
    # matches = \
    # re.findall(r'(?<![a-zA-Z0-9])([-‑]?\d+\.?\d*)(?=\.|\s|$)', str(text))

    # Convert to floats and ensure values are limited to 6 significant digits
    numeric_values = []
    for num in matches:
        num = num.replace('‑', '-')
        value = np.float64(num)  # Convert to NumPy float64 for consistent

        # Limit the value to 6 significant digits
        if value.is_integer(
        ):  # If it's an integer, format as an integer with 6 digits max
            value = f'{int(value):.6g}'
        else:  # For floats, format with 6 significant digits
            value = f'{value:.6g}'

        numeric_values.append(
            float(value))  # Convert back to float for numeric operations

    return numeric_values


RNA_CLASSES = sorted([
    '5S_rRNA', '5_8S_rRNA', 'tRNA', 'ribozyme', 'CD-box', 'miRNA',
    'Intron_gpI', 'Intron_gpII', 'HACA-box', 'riboswitch', 'IRES', 'leader',
    'scaRNA'
],
                     key=len,
                     reverse=True)

modification_classes = [
    'AtoI', 'm6Am', 'm1A', 'm5C', 'm5U', 'm6A', 'm7G', 'Psi', 'Am', 'Cm', 'Gm',
    'Um', 'none'
]


def generic_replace(m):
    candidate = m.group(1)

    if len(candidate) >= 4:
        # print(candidate)
        return f'<SMILES> {candidate} </SMILES>'
    else:
        return candidate


# Use the sentiment analysis model as fallback
# if classification by keywords fails
def classify_by_sentiment_model(text):
    text = [
        str(t).replace('</s>', '').replace('<pad>', '').strip() for t in text
    ]

    candidate_labels = [
        'Yes,I can positively identify', 'No,My answer is negative',
        'This protein is expected to dissolve in water',
        'This protein is not expected to dissolve in water'
    ]
    outputs = classifier(text, candidate_labels, batch_size=64)
    processed_results = []
    for output in outputs:
        # Hugging Face zero-shot pipeline默认按分数高低排序返回结果
        top_label = output['labels'][0]
        top_score = output['scores'][0]

        if (top_label == 'Yes,I can positively identify' or top_label
                == 'This protein is expected to dissolve in water'):
            result_class = 1
        else:
            result_class = 0

        processed_results.append((result_class, top_score))
    return processed_results


def classify_by_keywords(text):
    positive_keywords = [
        'Yes', 'yes', 'positive', 'Positive', 'empirical', 'plausible',
        'confirms', 'have detected', 'are discernible', 'are supported',
        'is supported', 'display', 'detected the presence', 'shows evidence',
        'has been identified', 'shows', 'has identified', 'contains ',
        'exhibits evidence', 'is plausible', 'contains identifiable', 'Indeed',
        'reveals the presence', 'include', 'are present', 'definitely has',
        'soluble', 'displays regions', 'has a high solubility',
        'dissolves easily', 'Solubility is expected',
        'is expected to dissolve', 'is predicted', 'is likely', 'is expected',
        'is expected to dissolve', 'will dissolve', 'dissolves easily'
    ]

    negative_keywords = [
        'No', 'no', 'negative', 'Negative', 'insoluble', 'does not',
        'unlikely', 'absence', 'not found', 'not detected', 'not associated',
        'not inferred', 'not linked', 'does not indicate', 'no evidence',
        'not predicted', 'absent', 'not present', 'no indicators',
        'not exhibit', 'are absent', 'found none', 'did not reveal', 'lacks',
        'exhibits no', 'insolubility', 'low solubility', 'not soluble',
        'not be soluble', 'does not display regions', 'cannot confirm'
    ]

    dont_know_keywords = [
        'don\'t know', 'unknown', 'unsure', 'uncertain', 'not applicable',
        'cannot confirm'
    ]

    text_lower = text.lower()

    # 为了安全，转义关键词中的特殊字符，并用'|'（或）连接
    # \b确保匹配的是整个单词
    negative_pattern = r'\b(' + '|'.join(
        re.escape(kw) for kw in negative_keywords) + r')\b'
    positive_pattern = r'\b(' + '|'.join(
        re.escape(kw) for kw in positive_keywords) + r')\b'
    dont_know_pattern = r'\b(' + '|'.join(
        re.escape(kw) for kw in dont_know_keywords) + r')\b'

    # 1. 检查负面关键词
    if re.search(negative_pattern, text_lower):
        return 0
    # 2. 检查正面关键词
    elif re.search(positive_pattern, text_lower):
        return 1
    # 3. 检查 "不知道" 关键词
    elif re.search(dont_know_pattern, text_lower):
        return 'dont_know'
    else:
        return None


# Save the processed data for each task in a separate file
def save_processed_data(model_name, task_name, task_processed_data):
    dir_path = path_bioinstruction + f'/processed_data/{model_name}'
    file_path = f'{dir_path}/{task_name}_processed_data.json'
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'w') as outfile:
        json.dump(task_processed_data, outfile, indent=4)

    print(f'Task {task_name} procssed data saved in {file_path}')


# Process regression task
def process_regression_task(task_name, task_entries, model_name):
    result_values = []
    label_values = []
    task_processed_data = []
    over_len = 0
    miss_len = 0
    for index, entry in enumerate(task_entries):
        # print(entry)
        if '<summary>' in entry['model_output']:
            entry['model_output'] = entry['model_output'].split(
                '<summary>')[-1]
        if '</think>' in entry['model_output']:
            entry['model_output'] = entry['model_output'].split('</think>')[-1]
            extracted_result = extract_numeric_values(entry['model_output'])
        else:
            if '<think>' in entry['model_output']:
                over_len += 1
                extracted_result = []
            else:
                miss_len += 1
                extracted_result = extract_numeric_values(
                    entry['model_output'])

        label = float(entry['label'])
        print('label', label)
        print('extracted_result', extracted_result)

        if len(extracted_result
               ) != 0 and extracted_result[0] > 80 and task_name == 'Isoform':
            print(entry['model_output'])
            extracted_result = []

        if len(extracted_result) != 1:
            print('not one:', entry['model_output'])
            extracted_result = []

        if len(extracted_result) == 0:
            result_values.append(
                np.inf)  # Assign infinity if no valid result is extracted
        else:
            result_values.append(
                extracted_result[0])  # Take the first valid extracted result

        label_values.append(label)

        task_processed_data.append({
            'input':
            entry['input'],
            'label':
            entry['label'],
            'processed_model_ouput':
            extracted_result[0] if len(extracted_result) > 0 else np.inf,
            'original_model_output':
            entry['model_output'],
        })

    save_processed_data(model_name, task_name, task_processed_data)
    print('over_len: ', over_len)
    print('miss_len: ', miss_len)
    return label_values, result_values


# Compute spearman correlation
def compute_spearman(label_values, result_values):
    if len(result_values) == 0:
        return {'spearman': 'Error: Empty data'}
    elif len(result_values) != len(label_values):
        return {
            'spearman':
            'Error: Mismatch in the number of extracted numeric values'
        }

    # Convert the label and result values to numpy arrays
    result_values = np.array(result_values).flatten()
    label_values = np.array(label_values).flatten()

    # Identify explicitly assigned infinity values
    near_infinity_mask = np.isinf(result_values)

    # Exclude near-infinity pairs from the main calculation
    valid_mask = ~near_infinity_mask & np.isfinite(
        result_values) & np.isfinite(label_values)
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    outlier_mask = valid_result_values <= 300

    valid_result_values = valid_result_values[outlier_mask]
    valid_label_values = valid_label_values[outlier_mask]

    # 初始化指标
    spearman = 0.0
    rmse = 0.0

    # Compute Spearman correlation for valid values
    if len(valid_result_values) > 0:
        spearman, _ = spearmanr(valid_label_values, valid_result_values)
        mse = mean_squared_error(valid_label_values, valid_result_values)
        # 然后开方得到 RMSE
        rmse = np.sqrt(mse)

    else:
        spearman = 0  # Fallback if no valid pairs

    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()
    num_infinity_values = near_infinity_mask.sum()

    if num_infinity_values > 0:
        final_spearman_score = (spearman * total_valid_points +
                                0 * num_infinity_values) / total_data_points
    else:
        final_spearman_score = spearman  # Edge case: no near-infinity values
    print('rmse:', rmse)

    return {'spearman': final_spearman_score}


# Compute R2
def compute_R2(label_values, result_values):
    # from sklearn.metrics import r2_score

    # y_true = np.asarray(label_values, dtype=float).flatten()
    # y_pred = np.asarray(result_values, dtype=float).flatten()

    # Check for empty data
    if len(result_values) == 0:
        return {'R2': 'Error: Empty data.'}

    # Check for equal length of arrays
    elif len(result_values) != len(label_values):
        return {
            'R2': 'Error: Mismatch in the number of extracted numeric values.'
        }

    # Convert the label and result values to numpy arrays
    result_values = np.array(result_values).flatten()
    label_values = np.array(label_values).flatten()

    # Identify explicitly assigned infinity values
    near_infinity_mask = np.isinf(result_values)

    # Exclude near-infinity pairs from the main calculation
    valid_mask = ~near_infinity_mask & np.isfinite(
        result_values) & np.isfinite(label_values)
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    # Compute Pearson correlation coefficient for valid values
    if len(valid_result_values) > 0:
        try:
            pcc, _ = pearsonr(valid_label_values, valid_result_values)
            R2 = pcc**2
            # mse = mean_squared_error(valid_label_values, valid_result_values)
            # 然后开方得到 RMSE
            # rmse = np.sqrt(mse)
        except Exception:
            R2 = np.inf  # Fallback to inf if computation fails
    else:
        R2 = 0  # Fallback if no valid pairs

    # Combine R2 score for valid and infinity values
    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()
    num_infinity_values = near_infinity_mask.sum()

    if num_infinity_values > 0:
        final_R2_score = (R2 * total_valid_points +
                          0 * num_infinity_values) / total_data_points
    else:
        final_R2_score = R2  # Edge case: no near-infinity values
    # print("RMSE:",rmse)
    return {'R2': final_R2_score}


# Compute mixed score
def compute_mixed_score(label_values,
                        result_values,
                        threshold=30,
                        max_value=1e3):
    rmse = 0.0
    if len(result_values) == 0:
        return {'mixed_score': 'Error: Empty data.'}
    elif len(result_values) != len(label_values):
        return {
            'mixed_score':
            'Error: Mismatch in the number of extracted numeric values'
        }

    # Convert the label and result values to numeric arrays
    # using pandas to handle non-numeric entries
    result_values = pd.to_numeric(result_values, errors='coerce').flatten()
    label_values = pd.to_numeric(label_values, errors='coerce').flatten()

    # Identify near-infinity values
    near_infinity_mask = np.abs(result_values) > max_value
    if near_infinity_mask.any():
        print(
            f'Warning: Found {sum(near_infinity_mask)} result values too large'
            ' will be assigned a mixed score of 0. '
            f'Large result values: {result_values[near_infinity_mask]} ')

    # Exclude near-infinity pairs from the main calculation
    valid_mask = ~near_infinity_mask & np.isfinite(
        result_values) & np.isfinite(label_values)
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    # Assign a mixed score of 0 to near-infinity pairs
    num_infinity_values = near_infinity_mask.sum()
    if num_infinity_values > 0:
        mixed_score_infinity = 0

    # Convert to binary based on the threshold for valid values
    label_binary = (valid_label_values < threshold).astype(int)
    result_binary = (valid_result_values < threshold).astype(int)

    # Compute precision, recall, F1 score for valid values
    precision = precision_score(label_binary, result_binary, average='binary')
    recall = recall_score(label_binary, result_binary, average='binary')
    f1 = 2 * precision * recall / (precision + recall) if (precision +
                                                           recall) != 0 else 0

    try:
        # Compute mean absolute error (MAE) for valid values
        mae = mean_absolute_error(valid_label_values, valid_result_values)
        mse = mean_squared_error(valid_label_values, valid_result_values)
        rmse = np.sqrt(mse)

    except ValueError:
        mae = np.inf  # Fallback to infinity if error occurs

    # Mask to keep only values in the range [0, threshold] for valid values
    mask = (valid_result_values >= 0) & (valid_result_values <= threshold)
    if mask.sum() > 0:
        range_mae = mean_absolute_error(valid_label_values[mask],
                                        valid_result_values[mask])
    else:
        range_mae = 100  # Fallback if no values within the range

    # Ensure MAE and range_mae are within reasonable bounds to avoid overflow
    mae = min(mae, 100)
    range_mae = min(range_mae, 100)

    # Compute mixed score for valid values
    mixed_score_valid = (1 - mae / 100) * 0.5 + (1 -
                                                 range_mae / 100) * f1 * 0.5
    print(
        f'(1 - mae / 100) * 0.5={(1 - mae / 100) * 0.5}\n '
        f'(1 - range_mae / 100)={(1 - range_mae / 100)}\n '
        f'(1 - range_mae / 100) * f1 * 0.5={(1 - range_mae / 100) * f1 * 0.5}')

    # Compute the final mixed score,
    # averaging in the score for the near-infinity pairs
    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()

    if num_infinity_values > 0:
        final_mixed_score = (
            mixed_score_valid * total_valid_points +
            mixed_score_infinity * num_infinity_values) / total_data_points
    else:
        # Edge case: no near-infinity values
        final_mixed_score = mixed_score_valid
    print('RMSE', rmse)

    return {'mixed_score': final_mixed_score}


# Programmable Switch task:
# multilabel regression output one average correlation
def compute_R2_for_ProgrammableRNASwitches_task(task_name, task_entries,
                                                model_name):
    on_result_values = []
    off_result_values = []
    on_off_result_values = []

    on_label_values = []
    off_label_values = []
    on_off_label_values = []

    task_processed_data = []
    over_len = 0
    miss_len = 0
    # Loop through each entry in the task
    for entry in task_entries:
        label = entry['label']
        # label = ast.literal_eval(label)
        on_label = float(label['ON'])
        off_label = float(label['OFF'])
        on_off_label = float(label['ON_OFF'])

        # Extract numeric values from the model output
        if '</think>' in entry['model_output']:
            entry['model_output'] = entry['model_output'].split('</think>')[-1]
        else:
            if '<think>' in entry['model_output']:
                over_len += 1
            else:
                miss_len += 1
        extracted_result = extract_numeric_values(entry['model_output'])
        print('extracted_result', extracted_result)

        # Handle missing or invalid data by assigning np.nan
        if len(extracted_result) != 3:
            on_result_values.append(np.nan)
            off_result_values.append(np.nan)
            on_off_result_values.append(np.nan)
        else:
            on_result = extracted_result[0]
            off_result = extracted_result[1]
            on_off_result = extracted_result[2]
            on_result_values.append(on_result)
            off_result_values.append(off_result)
            on_off_result_values.append(on_off_result)

        # Append the label values
        on_label_values.append(on_label)
        off_label_values.append(off_label)
        on_off_label_values.append(on_off_label)

        # Save processed task data for this entry
        task_processed_data.append({
            'input':
            entry['input'],
            'label':
            entry['label'],
            'processed_model_output': {
                'ON': on_result if len(extracted_result) == 3 else np.nan,
                'OFF': off_result if len(extracted_result) == 3 else np.nan,
                'ON_Off':
                on_off_result if len(extracted_result) == 3 else np.nan
            },
            'original_model_output':
            entry['model_output']
        })

    # Save the processed task data
    save_processed_data(model_name, task_name, task_processed_data)

    # Convert to numpy arrays for easier manipulation
    on_result_values = np.array(on_result_values)
    off_result_values = np.array(off_result_values)
    on_off_result_values = np.array(on_off_result_values)

    on_label_values = np.array(on_label_values)
    off_label_values = np.array(off_label_values)
    on_off_label_values = np.array(on_off_label_values)

    # Filter out NaN values in ON, OFF, and ON/OFF result/label pairs
    on_valid_mask = np.isfinite(on_result_values) & np.isfinite(
        on_label_values)
    off_valid_mask = np.isfinite(off_result_values) & np.isfinite(
        off_label_values)
    on_off_valid_mask = np.isfinite(on_off_result_values) & np.isfinite(
        on_off_label_values)

    # Filter the valid ON, OFF, and ON/OFF values
    on_result_values = on_result_values[on_valid_mask]
    off_result_values = off_result_values[off_valid_mask]
    on_off_result_values = on_off_result_values[on_off_valid_mask]

    on_label_values = on_label_values[on_valid_mask]
    off_label_values = off_label_values[off_valid_mask]
    on_off_label_values = on_off_label_values[on_off_valid_mask]

    try:
        on_R2 = compute_R2(
            on_result_values,
            on_label_values)['R2'] if len(on_result_values) > 0 else 0
    except Exception:
        on_R2 = 0  # Assign 0 in case of error

    try:
        off_R2 = compute_R2(
            off_result_values,
            off_label_values)['R2'] if len(off_result_values) > 0 else 0
    except Exception:
        off_R2 = 0  # Assign 0 in case of error

    try:
        on_off_R2 = compute_R2(
            on_off_result_values,
            on_off_label_values)['R2'] if len(on_off_result_values) > 0 else 0
    except Exception:
        on_off_R2 = 0  # Assign 0 in case of error

    # Combine R2 scores for ON, OFF, and ON/OFF values
    total_on_points = max(len(on_result_values) + np.sum(~on_valid_mask), 1)
    total_off_points = max(len(off_result_values) + np.sum(~off_valid_mask), 1)
    total_on_off_points = max(
        len(on_off_result_values) + np.sum(~on_off_valid_mask), 1)

    # Assign average R2 with 0 for invalid entries
    final_on_R2 = (on_R2 * len(on_result_values)) / total_on_points if len(
        on_result_values) > 0 else 0
    final_off_R2 = (off_R2 * len(off_result_values)) / total_off_points if len(
        off_result_values) > 0 else 0
    final_on_off_R2 = (on_off_R2 *
                       len(on_off_result_values)) / total_on_off_points if len(
                           on_off_result_values) > 0 else 0

    avg_R2 = (final_on_R2 + final_off_R2 + final_on_off_R2) / 3
    print('over_len: ', over_len)
    print('miss_len: ', miss_len)
    print('123', final_on_R2, final_off_R2, final_on_off_R2)
    return {'R2': avg_R2}


# Enhancer Activity Task:
# multilabel regression output two individual correlation
def compute_PCC_for_enhancer_activity_task(task_name, task_entries,
                                           model_name):
    hk_result_values = []
    dev_result_values = []

    hk_label_values = []
    dev_label_values = []

    task_processed_data = []
    over_len = 0
    miss_len = 0
    # Loop through each entry in the task
    for entry in task_entries:
        label = entry['label']
        # label = ast.literal_eval(label)
        if '</think>' in entry['model_output']:
            entry['model_output'] = entry['model_output'].split('</think>')[-1]
        else:
            if '<think>' in entry['model_output']:
                over_len += 1
            else:
                miss_len += 1
        model_output = entry['model_output']
        print('model_output', model_output)
        hk_label = float(label['hk'])
        dev_label = float(label['dev'])

        # Extract model output values for HK and Dev enhancer activity
        extracted_result = extract_numeric_values(model_output)

        # Handle missing or invalid data by assigning np.inf
        if len(extracted_result) != 2:

            hk_result_values.append(np.inf)
            dev_result_values.append(np.inf)
        else:
            hk_result = extracted_result[0]
            dev_result = extracted_result[1]
            hk_result_values.append(hk_result)
            dev_result_values.append(dev_result)

        # Append the label values
        hk_label_values.append(hk_label)
        dev_label_values.append(dev_label)

        # Save processed task data for this entry
        task_processed_data.append({
            'input':
            entry['input'],
            'label':
            entry['label'],
            'processed_model_output': {
                'hk': hk_result if len(extracted_result) == 2 else np.inf,
                'dev': dev_result if len(extracted_result) == 2 else np.inf
            },
            'original_model_output':
            entry['model_output']
        })

    # Save the processed task data
    save_processed_data(model_name, task_name, task_processed_data)

    # Convert to numpy arrays for easier manipulation
    hk_result_values = np.array(hk_result_values)
    dev_result_values = np.array(dev_result_values)
    hk_label_values = np.array(hk_label_values)
    dev_label_values = np.array(dev_label_values)

    # Filter out NaN or inf values in both HK and Dev result/label pairs
    hk_valid_mask = np.isfinite(hk_result_values) & np.isfinite(
        hk_label_values)
    dev_valid_mask = np.isfinite(dev_result_values) & np.isfinite(
        dev_label_values)

    # Filter the valid HK and Dev values
    hk_result_values = hk_result_values[hk_valid_mask]
    hk_label_values = hk_label_values[hk_valid_mask]
    dev_result_values = dev_result_values[dev_valid_mask]
    dev_label_values = dev_label_values[dev_valid_mask]

    # Compute Pearson correlation for valid HK and Dev enhancer activities
    if len(hk_result_values) > 0:
        try:
            hk_pcc, _ = pearsonr(hk_result_values, hk_label_values)
        except Exception:
            hk_pcc = np.inf  # Set to inf in case of errors
    else:
        return {
            'PCC':
            'Error: HK has insufficient valid data '
            'after removing NaNs and infs.'
        }
    if len(dev_result_values) > 0:
        try:
            dev_pcc, _ = pearsonr(dev_result_values, dev_label_values)
        except Exception:
            dev_pcc = np.inf  # Set to inf in case of errors
    else:
        return {
            'PCC':
            'Error: Dev has insufficient valid data '
            'after removing NaNs and infs.'
        }

    # Combine results with NaN/inf values consideration
    total_hk_points = len(hk_result_values) + np.sum(~hk_valid_mask)
    total_dev_points = len(dev_result_values) + np.sum(~dev_valid_mask)

    # Assign mixed score with 0 for invalid entries
    final_hk_pcc = (hk_pcc * len(hk_result_values) + 0 * np.sum(~hk_valid_mask)
                    ) / total_hk_points if len(hk_result_values) > 0 else 0
    final_dev_pcc = (dev_pcc * len(dev_result_values) +
                     0 * np.sum(~dev_valid_mask)) / total_dev_points if len(
                         dev_result_values) > 0 else 0
    print('over_len:', over_len)
    print('miss_len: ', miss_len)
    return {
        'PCC': (final_hk_pcc + final_dev_pcc) / 2,
        'hk_PCC': final_hk_pcc,
        'dev_PCC': final_dev_pcc
    }


# Process binary classification task
def process_binary_classification_task(task_name, task_entries, model_name):
    label_classes = []
    result_classes = []
    task_processed_data = []
    entries_for_model = []
    over_len = 0
    miss_len = 0
    for index, entry in enumerate(tqdm(task_entries)):
        if '<summary>' in entry['model_output']:
            entry['model_output'] = entry['model_output'].split(
                '<summary>')[-1]

        if '</think>' in entry['model_output']:
            entry['model_output'] = entry['model_output'].split('</think>')[-1]
        else:
            if '<think>' in entry['model_output']:
                over_len += 1
            else:
                miss_len += 1

        label_class = 1 if entry['label'] == 'positive' else 0
        model_output = entry['model_output']
        model_output = str(entry['model_output'])
        result_class = None
        score = 0

        if model_output is None:
            result_class = 1 - label_class
        else:
            keyword_result = classify_by_keywords(model_output)
            if keyword_result == 'dont_know':
                result_class = 1 - label_class
            elif keyword_result is not None:
                result_class = keyword_result
            else:
                if model_output and model_output.strip():
                    entries_for_model.append({
                        'index': index,
                        'text': model_output
                    })
                else:
                    result_class = 1 - label_class

        # 将已经处理完的条目先存起来，留出空位给模型处理结果
        task_processed_data.append({
            'input': entry['input'],
            'original_label': entry['label'],
            'processed_label': label_class,
            'original_model_output': model_output,
            'processed_model_output': result_class,  # 可能为None，后面会填充
            'score': 'N/A'  # 默认为N/A
        })
    print(len(entries_for_model))

    if entries_for_model:

        texts_to_classify = [item['text'] for item in entries_for_model]

        # 一次性将所有文本传给模型
        model_results = classify_by_sentiment_model(texts_to_classify)

        for i, model_item in enumerate(tqdm(entries_for_model)):
            original_index = model_item['index']
            result_class, score = model_results[i]

            # (可选逻辑) 如果置信度低，则判错
            # if score < 0.5:
            #     result_class =
            # 1 - task_processed_data[original_index]['processed_label']

            # 将模型处理的结果填回到最终数据列表的正确位置
            task_processed_data[original_index][
                'processed_model_output'] = result_class
            task_processed_data[original_index]['score'] = str(score)

    result_classes = [d['processed_model_output'] for d in task_processed_data]
    label_classes = [d['processed_label'] for d in task_processed_data]
    print('miss_len:', miss_len)
    print('over_len:', over_len)

    save_processed_data(model_name, task_name, task_processed_data)

    return label_classes, result_classes


# Compute matthews correlation coefficient (MCC)
def compute_MCC(label_classes, result_classes):
    if len(result_classes) == 0:
        return {'MCC': 'Error: Empty data.'}
    elif len(result_classes) != len(label_classes):
        return {
            'MCC': 'Error: Mismatch in the number of extracted numeric values.'
        }
    else:
        mcc = matthews_corrcoef(label_classes, result_classes)
        return {'MCC': mcc}


# Compute accuracy score (Acc)
def compute_Acc(label_classes, result_classes):
    if len(result_classes) == 0:
        return {
            'Acc':
            'Error: Insufficient data for classification. '
            'Number of model outputs is 0.'
        }
    elif len(result_classes) != len(label_classes):
        return {
            'Acc':
            'Error: Mismatched labels. '
            'The number of model outputs does not match the number of labels.'
        }
    else:
        acc = accuracy_score(label_classes, result_classes)
        return {'Acc': acc}


# Extract RNA family from the text
def extract_rna_family(text):
    for rna_class in RNA_CLASSES:
        if rna_class in text:
            return rna_class
    return None


# Compute ACC metric for NoncodingRNAFamily multiclass classification task
def compute_Acc_for_NoncodingRNAFamily_task(task_name, task_entries,
                                            model_name):
    correct_count = 0
    total_count = 0
    task_processed_data = []
    over_len = 0
    miss_len = 0
    for entry in task_entries:
        if '</think>' in entry['model_output']:
            entry['model_output'] = entry['model_output'].split('</think>')[-1]
            result_family = extract_rna_family(entry['model_output'])
        else:
            if '<think>' in entry['model_output']:
                over_len += 1
            else:
                miss_len += 1
            # result_family = "None"
            result_family = extract_rna_family(entry['model_output'])

        label_family = entry['label']
        # result_family = extract_rna_family(entry["model_output"])
        # Compare extracted family with the ground truth label
        if result_family == label_family:
            correct_count += 1

        total_count += 1

        # Store original and processed data
        task_processed_data.append({
            'input':
            entry['input'],
            'label':
            entry['label'],
            'processed_model_output':
            result_family,
            'original_model_output':
            entry['model_output']
        })

    save_processed_data(model_name, task_name, task_processed_data)
    print('over_len:', over_len)
    print('miss_len:', miss_len)
    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    if (total_count - over_len) != 0:
        print('true_acc:', correct_count / (total_count - over_len))

    return {'Acc': accuracy}


# Extract RNA modification labels from the output text
def extract_modifications(text):
    extracted_modifications = []
    for mod_class in modification_classes:
        # Use word boundaries to ensure whole-word match
        if re.search(rf'\b{mod_class}\b', text):
            extracted_modifications.append(mod_class)
    return extracted_modifications


# Convert modification labels to a binary multihot vector
def convert_to_binary_vector(modifications, classes=modification_classes):
    binary_vector = []

    # Handle case where modifications is None
    if modifications is None:
        modifications = []  # Treat None as an empty list

    for mod in classes:
        if mod in modifications:
            binary_vector.append(1)
        else:
            binary_vector.append(0)
    return binary_vector


# Compute AUC metrics for Modification task
def compute_AUC_for_Modification_task(task_name, task_entries, model_name):
    y_true = []
    y_pred = []
    task_processed_data = []
    over_len = 0
    miss_len = 0
    for entry in task_entries:
        # MARK:gaile
        if '<summary>' in entry['model_output']:
            entry['model_output'] = entry['model_output'].split(
                '<summary>')[-1]
        if '</think>' in entry['model_output']:
            entry['model_output'] = entry['model_output'].split('</think>')[-1]
        else:
            if '<think>' in entry['model_output']:
                over_len += 1
            else:
                miss_len += 1
        predicted_modifications = extract_modifications(entry['model_output'])
        # print(predicted_modifications)
        true_modifications = entry['label'].split(',')

        # Handle case where result is empty and label is "none"
        if not predicted_modifications:
            # Classify by keyword
            predicted_modifications = classify_by_keywords(
                entry['model_output'])

            # If keyword negative,
            # assigned to prediction to be the "none" class
            if predicted_modifications == 0:
                predicted_modifications = ['none']

            elif predicted_modifications == 1:
                predicted_modifications = []

            # If the result cannot be classified, use the sentiment model
            elif predicted_modifications is None:

                sentiment_result, sentiment_score = \
                    classify_by_sentiment_model(
                        [entry['model_output']])[0]

                # If classified as negative, manually label as 'none'
                if sentiment_result == 0:
                    predicted_modifications = ['none']

                else:
                    predicted_modifications = []

        # Convert the predicted and true modifications to binary vectors
        y_true.append(convert_to_binary_vector(true_modifications))
        y_pred.append(convert_to_binary_vector(predicted_modifications))

        # Store the processed data
        task_processed_data.append({
            'input':
            entry['input'],
            'label':
            entry['label'],
            'processed_model_ouput':
            predicted_modifications,
            'original_model_output':
            entry['model_output']
        })
        print('label', entry['label'])
        print('predication', predicted_modifications)

    save_processed_data(model_name, task_name, task_processed_data)
    print('over_len:', over_len)
    print('miss_len: ', miss_len)
    # Compute the AUC for each class, then average the AUC across all classes
    try:
        auc = roc_auc_score(y_true, y_pred, average='macro')
        print('auc', auc)
    except ValueError:
        auc = None

    return {'AUC': auc}


# FunctionEC Task
# Modified from
# SaProt https://github.com/westlake-repl/SaProt/blob/main/utils/metrics.py
def count_f1_max(pred, target):
    """
    F1 score with the optimal threshold.
    Handles cases where either predictions or targets are empty.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`

    Returns:
        float: The maximum F1 score or 0.0 if inputs are empty.
    """
    # Check if either pred or target is empty
    if pred.numel() == 0 or target.numel() == 0:
        return 0.0

    # Proceed with the original logic if inputs are not empty
    order = pred.argsort(descending=True, dim=1, stable=True)
    # print(f"order: {order}")
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)

    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)
    all_order = pred.flatten().argsort(descending=True, stable=True)
    order = order + torch.arange(
        order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]

    precision = precision.flatten()
    recall = recall.flatten()

    all_precision = precision[all_order] - \
        torch.where(
            is_start, torch.zeros_like(precision),
            precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
        torch.where(
            is_start, torch.zeros_like(recall),
            recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall +
                                               1e-10)

    if torch.isnan(all_f1).any():
        return 0.0

    return all_f1.max()


def round_and_scale_results(data, decimal_places=3, scale_factor=100):
    for key, value in data.items():
        if isinstance(value, dict):
            # Recursive call if the value is a dictionary
            round_and_scale_results(value, decimal_places, scale_factor)
        elif isinstance(value, (float, int)):
            # Round and scale numeric values
            data[key] = float(round(value * scale_factor, decimal_places))


# Convert EC number to binary multihot vectors
def ec_to_multihot(ec_list, ec_labels):
    multihot = torch.zeros(len(ec_labels))
    if not ec_list:  # Check if ec_list is empty
        return multihot
    multihot = torch.zeros(len(ec_labels))
    for ec in ec_list:
        if ec in ec_labels:
            idx = ec_labels.index(ec)
            multihot[idx] = 1
    return multihot


# Compute Fmax metric for FunctionEC task
def compute_Fmax_for_FunctionEC_task(task_name, task_entries, ec_labels,
                                     model_name):
    all_preds = []
    all_labels = []
    task_processed_data = []
    over_len = 0
    miss_len = 0
    for entry in task_entries:
        if '</think>' in entry['model_output']:
            entry['model_output'] = entry['model_output'].split('</think>')[-1]
        else:
            if '<think>' in entry['model_output']:
                over_len += 1
            else:
                miss_len += 1
        if '<summary>' in entry['model_output']:
            entry['model_output'] = entry['model_output'].split(
                '<summary>')[-1]
        # Parse the EC numbers from 'output' and 'label'
        label_ec = re.findall(r'\d+\.\d+\.\d+\.\-?\d*', entry['label'])
        result_ec = re.findall(r'\d+\.\d+\.\d+\.\-?\d*',
                               str(entry['model_output']))

        # Convert EC numbers to multi-hot vectors
        pred_multihot = ec_to_multihot(result_ec, ec_labels)
        label_multihot = ec_to_multihot(label_ec, ec_labels)

        # Store the results
        all_preds.append(pred_multihot)
        all_labels.append(label_multihot)

        # Save processed task data
        task_processed_data.append({
            'input':
            entry['input'],
            'label':
            entry['label'],
            'processed_label':
            label_ec,
            'original_model_output':
            entry['model_output'],
            'processed_model_output':
            result_ec,
        })
        print('label_ec', label_ec)
        print('result_ec', result_ec)

    save_processed_data(model_name, task_name, task_processed_data)

    # # Stack the predictions and targets for batch processing
    all_preds = torch.stack(all_preds)
    all_labels = torch.stack(all_labels)
    print('miss_len: ', miss_len)
    print('over_len: ', over_len)
    # Compute the Fmax score
    try:
        fmax_score = count_f1_max(all_preds, all_labels)
    except ValueError:
        fmax_score = None

    return {'Fmax': fmax_score.item()}


def preprocess_input_data(input_file_path, prediction, mini_set=False):
    data = []
    # Open the input file and process each line

    with open(input_file_path, 'r') as f:
        data_in = json.load(f)
    if mini_set and len(data_in) > 150:
        import random
        random.seed(1024)
        data_in = random.sample(data_in, 150)
        random.seed()

    if len(prediction) == len(data_in):
        for index in range(len(data_in)):
            try:
                data_list = {}
                data_list['input'] = data_in[index]['input']
                data_list['output'] = data_in[index]['output']
                # Try to load the line as a JSON object

                data_list['model_output'] = prediction[index]
                data_list['label'] = data_in[index]['label']
                # data_list['label']=data_in[index]['label']

                data_list['task'] = data_in[index]['task']
                # data_list['task']=data_in[index]['task']
                data.append(data_list)
                # Ensure the parsed data is a dictionary
            except json.JSONDecodeError:
                print(f'Skipping invalid line: {data_in[index]}')
    else:
        print('len(prediction)!=len(data_in) !!!')

    df = pd.DataFrame(data)  # Convert to a DataFrame
    # df = pd.read_json(input_file_path, lines=True, encoding_errors="ignore")
    print(f'Number of data samples: {len(df)}')
    df.rename(columns={'result': 'model_output'}, inplace=True)
    print(df['task'])
    df['task'] = df['task'].replace('rna_protein_interaction',
                                    'ncRNAProteinInter')
    df['task'] = df['task'].replace('antibody_antigen', 'AntibodyAntigen')
    # Process entries with null labels
    # null_label_df = df[df['label'].isna()]
    # # null_label_df.to_json(f"{model_name}_result_label_null.json",
    # orient='records', lines=True)

    # Remove data for _all task
    # df = df[~df['task'].str.endswith('_all')]

    # Replace 'tf-h' with 'tf_h' and 'tf-m' with 'tf_m' in the 'task' column
    df['task'] = df['task'].str.replace('tf-h', 'tf_h')
    df['task'] = df['task'].str.replace('tf-m', 'tf_m')

    # Keep data if label is not null
    df = df[df['label'].notna()]
    df.reset_index(inplace=True, drop=True)

    # Convert to dictionary format for grouping
    data = df.to_dict(orient='records')

    # Group the data by 'task'
    grouped_data = defaultdict(list)
    for entry in data:
        task_name = entry['task'].split('-')[0]
        grouped_data[task_name].append(entry)

    return grouped_data


class bio_instruction_Evaluator(BaseEvaluator):

    def __init__(self, path, model_name, mini_set=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_path = path
        self.model_name = model_name
        self.mini_set = mini_set

    def score(self, predictions):
        test_path = self.dataset_path
        repo_id = '/'.join(test_path.split('/')[:-3])
        ec_path = 'ec_labels.json'
        ec_file_path = os.path.join(repo_id, ec_path)
        # ec_file_path = hf_hub_download(repo_id, ec_path, repo_type="dataset")

        with open(ec_file_path, 'r') as f:
            ec_labels = json.load(f)

        test_path = test_path.split(repo_id + '/')[1]
        input_file_path = self.dataset_path
        # input_file_path =
        # hf_hub_download(repo_id, test_path, repo_type="dataset")

        grouped_data = preprocess_input_data(input_file_path,
                                             predictions,
                                             mini_set=self.mini_set)

        print(f'Grouped data for tasks: {list(grouped_data.keys())}')

        register_tasks_path = 'register_tasks.json'
        register_tasks_file_path = os.path.join(repo_id, register_tasks_path)
        # register_tasks_file_path =
        # hf_hub_download(repo_id, register_tasks_path, repo_type="dataset")
        with open(register_tasks_file_path, 'r') as f:
            task_type_data = json.load(f)

        metrics = {}

        # Loop over tasks
        for task_name, task_entries in grouped_data.items():
            task_type = task_type_data[task_name]['type']
            task_metrics = task_type_data[task_name]['metrics']
            print(f'Prosessing {task_name} task...')
            print(task_type)
            sys.stdout.flush()

            if task_type == 'regression':
                # task_processed_data, label_values, result_values
                # = process_regression_task(task_name, task_entries)
                label_values, result_values = process_regression_task(
                    task_name, task_entries, self.model_name)
                if task_metrics == 'spearman':
                    metrics[task_name] = compute_spearman(
                        label_values, result_values)

                elif task_metrics == 'R2':
                    metrics[task_name] = compute_R2(label_values,
                                                    result_values)
                    # print(metrics[task_name])

                elif task_metrics == 'mixed_score':
                    metrics[task_name] = compute_mixed_score(label_values,
                                                             result_values,
                                                             threshold=30)

            elif task_type == 'binary classification':
                # task_processed_data, label_classes, result_classes
                # = process_binary_classification_task(task_name, task_entries)
                label_classes, result_classes = \
                    process_binary_classification_task(
                        task_name, task_entries, self.model_name)
                print(f'label_classes: {label_classes}')
                print(f'result_classes: {result_classes}')
                if task_metrics == 'MCC':
                    metrics[task_name] = compute_MCC(label_classes,
                                                     result_classes)

                elif task_metrics == 'Acc':
                    metrics[task_name] = compute_Acc(label_classes,
                                                     result_classes)

            elif task_type == 'multilabel regression':

                if task_name == 'ProgrammableRNASwitches':
                    metrics[task_name] = \
                        compute_R2_for_ProgrammableRNASwitches_task(
                            task_name, task_entries, self.model_name)

                elif task_name == 'enhancer_activity':
                    metrics[
                        task_name] = compute_PCC_for_enhancer_activity_task(
                            task_name, task_entries, self.model_name)

            elif task_type == 'multiclass classification':

                if task_name == 'NoncodingRNAFamily':
                    metrics[
                        task_name] = compute_Acc_for_NoncodingRNAFamily_task(
                            task_name, task_entries, self.model_name)

            elif task_type == 'multilabel classification':
                if task_name == 'FunctionEC':
                    metrics[task_name] = compute_Fmax_for_FunctionEC_task(
                        task_name, task_entries, ec_labels, self.model_name)

                elif task_name == 'Modification':
                    metrics[task_name] = compute_AUC_for_Modification_task(
                        task_name, task_entries, self.model_name)

            print(f'The metrics {task_metrics} for task {task_name}'
                  f' is {str(metrics[task_name][task_metrics])}')
            sys.stdout.flush()

        metrics_grouped_by_omics = defaultdict(dict)

        for task_name, task_metrics in metrics.items():
            # Get the omics type from task_type_data
            omics = task_type_data[task_name]['omics']

            # Scale the metrics
            scaled_metrics = task_metrics.copy(
            )  # Make a copy to avoid modifying the original
            round_and_scale_results(
                scaled_metrics)  # Apply scaling to the metrics

            # Add the scaled metrics to the grouped dictionary
            metrics_grouped_by_omics[omics][task_name] = scaled_metrics

            # Save the metrics (results) to a new JSON file
            metrics_file_path = (
                path_bioinstruction + f'/metrics_result/{omics}/' +
                f'metrics_result_{self.model_name}_{task_name}.json')
            output_directory = os.path.dirname(metrics_file_path)
            os.makedirs(output_directory, exist_ok=True)
            with open(metrics_file_path, 'w') as outfile:
                json.dump(metrics_grouped_by_omics[omics], outfile, indent=4)
            print(f'Metrics saved to {metrics_file_path}')

        return metrics_grouped_by_omics[omics][task_name]
