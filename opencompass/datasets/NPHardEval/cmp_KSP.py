import ast
import json

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .prompts import kspPrompts


def q2text(q, p=kspPrompts):
    knapsack_capacity = q['knapsack_capacity']
    items = q['items']
    prompt_text = p['Intro'] + '\n' + \
                p['Initial_question'].format(knapsack_capacity=knapsack_capacity) + '\n' + \
                p['Output_content'] + '\n' + \
                p['Output_format'] + \
                '\n The items details are as below: \n'
    for item in items:
        this_line = f"Item {item['id']} has weight {item['weight']} and value {item['value']}."
        prompt_text += this_line + '\n'
    return prompt_text


@LOAD_DATASET.register_module(force=True)
class CMP_KSP_Dataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        raw_data = []
        data_path = path
        all_data = []
        with open(data_path + 'ksp_instances.json', 'r') as f:
            data = json.load(f)
            for sample in data:
                level = len(sample['items']) - 3
                all_data.append((level, sample))
        for (level, q) in all_data:
            prompt = q2text(q)
            raw_data.append({
                'prompt': prompt,
                'q': str(level) + '####\n' + json.dumps(q),
                'level': level
            })
        dataset = Dataset.from_list(raw_data)
        return dataset


@ICL_EVALUATORS.register_module(force=True)
class CMP_KSP_Evaluator(BaseEvaluator):

    def score(self, predictions, references):
        assert len(predictions) == len(references)

        result = {'pass': 0, 'fail': 0}
        details = {}
        for index, (q, output) in enumerate(zip(references, predictions)):
            output_dict = {}
            level = int(q.split('####\n')[0])
            q = json.loads(q.split('####\n')[-1])
            try:
                llm_string = q
                output, reasoning = self.parse_xml_to_dict(llm_string)
                output_dict['output'] = output
                output_dict['correctness'], _ = self.kspCheck(q, output)
                output_dict['reasoning'] = reasoning
                output_dict['level'] = level
            except Exception as e:
                print(f'Attempt failed: {e}')
            if output_dict:
                if output_dict['correctness']:
                    r = 'pass'
                else:
                    r = 'fail'
            else:
                print(f'Failed to run {q}')
                r = 'fail'

            result[r] += level
            details[str(index)] = {'q': q, 'output': output, 'result': r}

        result['score'] = result['pass'] / (result['pass'] + result['fail']) * 100
        result['details'] = details
        final_result = {'Weighted Accuracy': result['score']}
        return final_result

    def parse_xml_to_dict(self, xml_string):
        try:
            assert '<final_answer>' in xml_string
            assert '</final_answer>' in xml_string
            assert '<reasoning>' in xml_string
            assert '</reasoning>' in xml_string
            final_answer_start = xml_string.index('<final_answer>') + len('<final_answer>')
            final_answer_end = xml_string.index('</final_answer>')
            reasoning_start = xml_string.index('<reasoning>') + len('<reasoning>')
            reasoning_end = xml_string.index('</reasoning>')
            final_answer_element = xml_string[final_answer_start:final_answer_end].rstrip().strip().rstrip()
            reasoning_element = xml_string[reasoning_start:reasoning_end].rstrip().strip().rstrip()
            try:
                final_answer_element = ast.literal_eval(final_answer_element)
            except Exception:
                final_answer_element = ''
        except Exception:
            final_answer_element = ''
            reasoning_element = ''

        return final_answer_element, reasoning_element

    def ksp_optimal_solution(self, knapsacks, capacity):
        """Provides the optimal solution for the KSP instance with dynamic
        programming.

        :param knapsacks: A dictionary of the knapsacks.
        :param capacity: The capacity of the knapsack.
        :return: The optimal value.
        """
        # num_knapsacks = len(knapsacks)

        # Create a one-dimensional array to store intermediate solutions
        dp = [0] * (capacity + 1)

        for itemId, (weight, value) in knapsacks.items():
            for w in range(capacity, weight - 1, -1):
                dp[w] = max(dp[w], value + dp[w - weight])

        return dp[capacity]

    # KSP
    def kspCheck(self, instance, solution):
        """Validates the solution for the KSP instance.

        :param instance: A dictionary of the KSP instance.
        :param solution: A dictionary of the solution.
        :return: A tuple of (is_correct, message).
        """
        # Change string key to integer key and value to boolean
        items = instance.get('items', [])
        knapsacks = {
            item['id']: (item['weight'], item['value'])
            for item in items
        }

        ksp_optimal_value = self.ksp_optimal_solution(
            knapsacks, instance['knapsack_capacity'])

        try:
            is_feasible = (solution.get('Feasible', '').lower() == 'yes')
        except Exception:
            return False, f'Output format is incorrect.'
        if is_feasible != (ksp_optimal_value > 0):
            return False, f'The solution is {is_feasible} but the optimal solution is {ksp_optimal_value > 0}.'

        total_value = int(solution.get('TotalValue', -1))
        selectedItems = list(map(int, solution.get('SelectedItemIds', [])))

        if len(set(selectedItems)) != len(selectedItems):
            return False, f'Duplicate items are selected.'

        total_weight = 0
        cum_value = 0

        # Calculate total weight and value of selected items
        for item in selectedItems:
            if knapsacks.get(item, False):
                weight, value = knapsacks[item]
                total_weight += weight
                cum_value += value
            else:
                return False, f'Item {item} does not exist.'

        # Check if the item weight exceeds the knapsack capacity
        if total_weight > instance['knapsack_capacity']:
            return False, f"Total weight {total_weight} exceeds knapsack capacity {instance['knapsack_capacity']}."

        if total_value != cum_value:
            return False, f'The total value {total_value} does not match the cumulative value {cum_value} of the selected items.'

        if total_value != ksp_optimal_value:
            return False, f'The total value {total_value} does not match the optimal value {ksp_optimal_value}.'

        return True, f'The solution is valid with total weight {total_weight} and total value {total_value}.'
