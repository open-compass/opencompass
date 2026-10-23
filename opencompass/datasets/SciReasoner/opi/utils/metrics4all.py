import argparse
import json
import os

import tqdm
from rouge_score import rouge_scorer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import MultiLabelBinarizer

from .accuracy4fold_type import compute_accuracy4fold_type


def calculate_metrics(output, target):
    # Convert to binary format
    mlb = MultiLabelBinarizer(classes=sorted(set(output + target)))
    y_true = mlb.fit_transform([target])
    y_pred = mlb.transform([output])

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true,
                                y_pred,
                                average='micro',
                                zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)

    return accuracy, precision, recall, f1


def calculate_rouge_l(output, target):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(' '.join(target), ' '.join(output))
    return scores['rougeL'].fmeasure


def process_json_file(json_file_path):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    rouge_ls = []

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    for entry in tqdm.tqdm(data):
        output = entry.get('output', entry.get('predict', []))
        target = entry.get('target', [])

        # Ensure both output and target are lists
        if isinstance(output, str):
            if any(keyword in json_file_path for keyword in
                   ['EC_number', 'go_terms', 'keywords', 'gene', 'domain']):
                output = output.split('; ')
            elif any(keyword in json_file_path
                     for keyword in ['function', 'subcell_loc', 'ss']):
                output = [output]
        if isinstance(target, str):
            if any(keyword in json_file_path for keyword in
                   ['EC_number', 'go_terms', 'keywords', 'gene', 'domain']):
                target = target.split('; ')
            elif any(keyword in json_file_path
                     for keyword in ['function', 'subcell_loc', 'ss']):
                target = [target]

        if 'function' in json_file_path:
            rouge_l = calculate_rouge_l(output, target)
            rouge_ls.append(rouge_l)
        elif 'subcell_loc' in json_file_path:
            accuracy, _, _, _ = calculate_metrics(output, target)
            accuracies.append(accuracy)
        else:
            _, precision, recall, f1 = calculate_metrics(output, target)
            # accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

    if 'function' in json_file_path:
        mean_rouge_l = sum(rouge_ls) / len(rouge_ls) if rouge_ls else 0
        return {'ROUGE-L': round(mean_rouge_l, 4)}, None
    elif 'subcell_loc' in json_file_path:
        mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        return {'Accuracy': round(mean_accuracy, 4)}, None
    else:
        mean_precision = sum(precisions) / len(precisions) if precisions else 0
        mean_recall = sum(recalls) / len(recalls) if recalls else 0
        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        return {
            'Precision': round(mean_precision, 4),
            'Recall': round(mean_recall, 4),
            'F1 Score': round(mean_f1, 4)
        }, None


def main(eval_res_path):
    results = {}

    # List all JSON files in the directory
    for file_name in sorted(os.listdir(eval_res_path)):
        if file_name.endswith('.json') and 'metrics_result' not in file_name:
            print(f'Processing {file_name}')
            file_path = os.path.join(eval_res_path, file_name)
            if 'function' in file_path:
                metrics, _ = process_json_file(file_path)
                results[file_name] = {'ROUGE-L': metrics['ROUGE-L']}
            elif 'subcell' in file_path:
                metrics, _ = process_json_file(file_path)
                results[file_name] = {'Accuracy': metrics['Accuracy']}
            elif 'fold_type' in file_path:
                test_files = [
                    'compute_scores/remote_homology_test_fold_holdout.json',
                    ('compute_scores/'
                     'remote_homology_test_superfamily_holdout.json'),
                    'compute_scores/remote_homology_test_family_holdout.json'
                ]
                acc_dict = compute_accuracy4fold_type(file_path, test_files)
                results[file_name] = acc_dict
            else:
                metrics, _ = process_json_file(file_path)
                results[file_name] = {
                    # 'Accuracy': metrics['Accuracy'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'F1 Score': metrics['F1 Score']
                }
            print(results[file_name])
    with open(f'{eval_res_path}/metrics_result.json', 'w') as result_file:
        json.dump(results, result_file, indent=4)

    print(f'Results saved to: {eval_res_path}/metrics_result.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir',
                        required=True,
                        help='Path to the result file dir')
    args = parser.parse_args()

    main(args.indir)
