import json
import os

import tqdm


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def compute_accuracy4fold_type(eval_file, test_files):
    """Compute accuracy for predictions against test datasets."""
    # Load evaluation data
    eval_data = load_json(eval_file)
    acc_dict = {}
    # Iterate over each test file
    for test_file in test_files:
        # Load test data
        test_data = load_json(test_file)

        # Create a set of test sequences
        test_seq_set = {item['primary'] for item in test_data}

        # Initialize counters
        correct_predictions = 0
        total_predictions = 0

        # Evaluate each item in the evaluation data
        for item in tqdm.tqdm(eval_data):
            if item['input'] not in test_seq_set:
                continue
            predict = item.get('output', item.get('predict', []))
            label = item['target']
            if predict == label:
                correct_predictions += 1
            total_predictions += 1

        # Calculate and print accuracy
        accuracy = correct_predictions / total_predictions \
            if total_predictions > 0 else 0
        acc_dict[os.path.basename(test_file).split('.')[0][21:]] = round(
            accuracy, 4)
    return acc_dict
