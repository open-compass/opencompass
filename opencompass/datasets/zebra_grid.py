import json
import re
from typing import Dict, List

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset

# ZeroEval's exact prompt template - fully aligned with official implementation
ZEBRA_GRID = """
# Example Puzzle 

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {
        "House 1": {
            "Name": "Arnold",
            "Drink": "tea"
        },
        "House 2": {
            "Name": "Peter",
            "Drink": "water"
        },
        "House 3": {
            "Name": "Eric",
            "Drink": "milk"
        }
    }
}

# Puzzle to Solve 

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

"""


@LOAD_DATASET.register_module()
class ZebraGridDataset(BaseDataset):
    """Zebra Grid Logic Puzzle Dataset.
    
    This dataset contains 1000 logic puzzle problems with grid-based solving.
    Each puzzle requires logical reasoning to determine relationships between entities.
    Based on allenai/ZebraLogicBench dataset implementation in ZeroEval.
    """

    @staticmethod
    def load(path: str = None):
        """Load the zebra grid dataset using HuggingFace datasets."""
        # Load data using HuggingFace datasets - cache directory should be set via environment variables
        dataset = load_dataset("allenai/ZebraLogicBench-private", "grid_mode", split="test")

        # Process data for OpenCompass format
        processed_data = []
        for idx, item in enumerate(dataset):
            # Comment out for full evaluation of 1000 samples
            # if idx >= 1:
            #     break
            
            # Add the ZeroEval-style prompt to the puzzle
            puzzle_with_prompt = apply_zebra_grid_template(item)
            
            # Create processed item with puzzle and solution
            processed_item = {
                'puzzle': puzzle_with_prompt,
                'solution': item['solution'],  # Keep solution for evaluation
                'id': item.get('id', str(idx)),
                'size': item.get('size', 'unknown')
            }
            processed_data.append(processed_item)

        # Create test split (all data is test data)
        dataset = Dataset.from_list(processed_data)
        return DatasetDict({'test': dataset})


def apply_zebra_grid_template(item):
    """Apply ZeroEval-style prompt template to zebra grid item."""
    
    # Use exact ZeroEval function logic
    prompt_str = ZEBRA_GRID[:]
    prompt_str = prompt_str.replace("{puzzle}", item["puzzle"])
    num_houses = len(item["solution"]["rows"])
    columns = item["solution"]["header"]
    assert columns[0] == "House"
    json_template = {"reasoning": "___", "solution": {}}
    for i in range(num_houses):
        json_template["solution"][f'House {i+1}'] = {columns[j]: "___" for j in range(1, len(columns))}
    json_str = json.dumps(json_template, indent=4)
    prompt_str = prompt_str.replace("{json_template}", json_str)
    return prompt_str


def extract_last_complete_json(s):
    """ZeroEval's exact JSON extraction function."""
    # Stack to keep track of opening and closing braces
    stack = []
    last_json_start = None
    last_json_str = None
    
    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
            if last_json_start is None:
                last_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    # Complete JSON object found
                    last_json_str = s[last_json_start:i+1]
                    last_json_start = None
    
    # Load the last JSON object
    if last_json_str:
        try:
            return json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            pass
    
    return None


class ZebraGridEvaluator(BaseEvaluator):
    """Evaluator for Zebra Grid dataset using puzzle accuracy metric.
    
    Based on ZeroEval's zebra_grid_eval.py implementation.
    """

    def __init__(self):
        super().__init__()

    def score(self, predictions: List[str], references: List[str], test_set: List[Dict] = None) -> Dict:
        """Calculate puzzle accuracy - exact match of the complete solution grid."""
        if len(predictions) != len(test_set):
            return {'error': 'predictions and test_set have different lengths'}

        if not test_set:
            return {'error': 'test_set is required for ZebraGrid evaluation'}

        solved_puzzles = 0
        total_puzzles = len(predictions)
        correct_cells = 0
        total_cells = 0
        no_answer = 0

        for i, (pred, test_item) in enumerate(zip(predictions, test_set)):
            # Get the raw solution from test_item
            raw_solution = test_item.get('solution', {})

            # Build solution table like ZeroEval
            solution_table = {}
            num_houses = len(raw_solution.get('rows', []))
            columns = raw_solution.get('header', [])

            if not columns or columns[0] != "House":
                continue

            this_total_cells = 0
            for house_idx in range(num_houses):
                house_key = f'House {house_idx + 1}'
                solution_table[house_key] = {}
                for col_idx in range(1, len(columns)):  # Skip "House" column
                    col_name = columns[col_idx]
                    cell_value = raw_solution['rows'][house_idx][col_idx]
                    solution_table[house_key][col_name] = cell_value
                    this_total_cells += 1

            total_cells += this_total_cells

            # Extract JSON from model prediction
            prediction_json = extract_last_complete_json(pred)

            if not prediction_json or 'solution' not in prediction_json:
                no_answer += 1
                continue

            prediction_table = prediction_json['solution']

            # Count correct cells following ZeroEval logic
            this_correct_cells = 0
            for house in solution_table:
                for column in solution_table[house]:
                    if house in prediction_table and column in prediction_table[house]:
                        truth_cell = solution_table[house][column].lower().strip()
                        predicted_cell = prediction_table[house][column]

                        if predicted_cell is None or len(str(predicted_cell)) == 0:
                            continue

                        if isinstance(predicted_cell, list):
                            predicted_cell = predicted_cell[0] if predicted_cell else ""

                        predicted_cell = str(predicted_cell).lower().strip()

                        if truth_cell == predicted_cell:
                            this_correct_cells += 1

            correct_cells += this_correct_cells

            # Check if puzzle is completely solved
            if this_correct_cells == this_total_cells:
                solved_puzzles += 1

        puzzle_accuracy = 100.0 * solved_puzzles / total_puzzles if total_puzzles > 0 else 0.0
        cell_accuracy = 100.0 * correct_cells / total_cells if total_cells > 0 else 0.0

        # Return only numeric metrics that OpenCompass summarizer expects
        return {
            'accuracy': puzzle_accuracy,  # Main metric OpenCompass will display
            'puzzle_accuracy': puzzle_accuracy,
            'cell_accuracy': cell_accuracy,
        }
