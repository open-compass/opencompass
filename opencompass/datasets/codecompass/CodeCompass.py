import json
import re
from typing import Any, Dict, Optional

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

try:
    from opencompass.utils import get_data_path

    from ..base import BaseDataset
except ImportError:

    class BaseDataset:
        pass

    def get_data_path(path, local_mode=False):
        return path


class CodeCompassCodeGenerationDataset(BaseDataset):

    DEFAULT_SYSTEM_PROMPT = (
        '[[Instruction]]\n'
        'You are an expert C++ programmer.'
        'You will be given a question in the format '
        'of an Online Judge (OJ) problem. You must generate a correct, '
        'self-contained C++ program that solves the problem and reads from '
        'standard input and writes to standard output. You will NOT return '
        'anything except for the program. Put your fixed program within code '
        'delimiters, for example:\n'
        '```cpp\n'
        '#include <iostream>\n'
        'using namespace std;\n\n'
        'int main() {\n'
        '    // YOUR CODE HERE\n'
        '    return 0;\n'
        '}\n'
        '```\n')

    DEFAULT_PROBLEM_TEMPLATE = """
[[Problem begin]]
{problem}
[[Problem end]]
"""

    @staticmethod
    def load(path: str = 'opencompass/CodeCompass',
             difficulty: Optional[str] = None,
             source: Optional[str] = None,
             system_prompt: Optional[str] = None,
             problem_template: Optional[str] = None) -> DatasetDict:

        try:
            raw_dataset = load_dataset(path=path,
                                       difficulty=difficulty,
                                       source=source,
                                       split='test',
                                       trust_remote_code=True)
            print(f'  > Raw dataset loaded: {len(raw_dataset)} samples')
        except Exception as e:
            print(f'Error loading dataset: {e}')
            raise

        if system_prompt is None:
            system_prompt = (
                CodeCompassCodeGenerationDataset.DEFAULT_SYSTEM_PROMPT)

        if problem_template is None:
            problem_template = (
                CodeCompassCodeGenerationDataset.DEFAULT_PROBLEM_TEMPLATE)

        processed_data = []
        failed_count = 0

        for item in tqdm(raw_dataset, desc='Processing samples'):
            try:
                processed_item = (
                    CodeCompassCodeGenerationDataset._process_item(
                        item, system_prompt, problem_template))
                if processed_item is not None:
                    processed_data.append(processed_item)
                else:
                    failed_count += 1
            except Exception as e:
                question_id = item.get('question_id', 'unknown')
                print(f'Error processing item {question_id}: {e}')
                failed_count += 1

        final_dataset = Dataset.from_list(processed_data)
        return DatasetDict({'test': final_dataset})

    @staticmethod
    def _extract_limits(problem_text: str) -> Dict[str, Any]:
        limits = {'time_limit_s': 2.0, 'memory_limit_mb': 256}

        time_match = re.search(r'Time Limit:\s*([0-9.]+)\s*s', problem_text,
                               re.IGNORECASE)
        if time_match:
            try:
                limits['time_limit_s'] = float(time_match.group(1))
            except ValueError:
                time_value = time_match.group(1)
                print(f'Warning: Could not parse time limit value: '
                      f'{time_value}')

        mem_match = re.search(r'Memory Limit:\s*(\d+)\s*MB', problem_text,
                              re.IGNORECASE)
        if mem_match:
            try:
                limits['memory_limit_mb'] = int(mem_match.group(1))
            except ValueError:
                mem_value = mem_match.group(1)
                print(f'Warning: Could not parse memory limit value: '
                      f'{mem_value}')

        return limits

    @staticmethod
    def _process_item(item: Dict[str, Any], system_prompt: str,
                      problem_template: str) -> Optional[Dict[str, Any]]:
        try:
            new_item = item.copy()

            problem_content = item.get('problem', '')
            if not problem_content:
                question_id = item.get('question_id')
                print(f'Warning: Empty problem for question_id {question_id}')
                return None

            full_prompt = system_prompt + problem_template.format(
                problem=problem_content)
            new_item['prompt'] = full_prompt

            evaluation_sample = (CodeCompassCodeGenerationDataset.
                                 _create_evaluation_sample(item))
            if evaluation_sample is None:
                question_id = item.get('question_id')
                print(f'Warning: Cannot create evaluation_sample for '
                      f'question_id {question_id}')
                return None

            new_item['evaluation_sample'] = evaluation_sample

            limits = CodeCompassCodeGenerationDataset._extract_limits(
                problem_content)

            eval_inputs = evaluation_sample.get('inputs', [])
            num_test_cases = (len(eval_inputs) if isinstance(
                evaluation_sample, dict) else 0)

            new_item['metadata'] = {
                'question_id': item.get('question_id'),
                'difficulty': item.get('difficulty', 'Unknown'),
                'source': item.get('source', 'Unknown'),
                'problem_length': len(problem_content),
                'num_test_cases': num_test_cases,
                'time_limit_s': limits['time_limit_s'],
                'memory_limit_mb': limits['memory_limit_mb']
            }

            new_item['original_problem'] = problem_content
            new_item['question_id'] = item.get('question_id')
            new_item['difficulty'] = item.get('difficulty')
            new_item['source'] = item.get('source')

            return new_item

        except Exception as e:
            print(f'Error in _process_item: {e}')
            return None

    @staticmethod
    def _create_evaluation_sample(
            item: Dict[str, Any]) -> Optional[Dict[str, Any]]:

        try:

            if 'cases' in item:
                cases_data = item['cases']

                if isinstance(cases_data, str):
                    cases_list = json.loads(cases_data)
                elif isinstance(cases_data, list):
                    cases_list = cases_data
                else:
                    print(f'Unknown cases format: {type(cases_data)}')
                    return None

                inputs = []
                outputs = []

                for case in cases_list:
                    if isinstance(case, dict):
                        input_val = case.get('input', '')
                        output_val = case.get('output', '')

                        if not isinstance(input_val, str):
                            input_val = str(input_val)
                        if not isinstance(output_val, str):
                            output_val = str(output_val)

                        inputs.append(input_val)
                        outputs.append(output_val)
                    else:
                        print(f'Warning: Invalid case format: {type(case)}')
                        continue

                if not inputs or not outputs:
                    print('Warning: No valid test cases found')
                    return None

                if len(inputs) != len(outputs):
                    input_count = len(inputs)
                    output_count = len(outputs)
                    print(f'Warning: Input/output count mismatch: '
                          f'{input_count} vs {output_count}')
                    return None

                return {
                    'inputs': inputs,
                    'outputs': outputs,
                    'fn_name': None,
                    'num_cases': len(inputs)
                }

            elif 'inputs' in item and 'outputs' in item:
                inputs = item['inputs']
                outputs = item['outputs']

                if not isinstance(inputs, list) or not isinstance(
                        outputs, list):
                    print('Warning: inputs/outputs are not lists')
                    return None

                if len(inputs) != len(outputs):
                    input_count = len(inputs)
                    output_count = len(outputs)
                    print(f'Warning: Input/output count mismatch: '
                          f'{input_count} vs {output_count}')
                    return None

                return {
                    'inputs': [str(inp) for inp in inputs],
                    'outputs': [str(out) for out in outputs],
                    'fn_name': None,
                    'num_cases': len(inputs)
                }

            else:
                print('Warning: No test cases found in item')
                return None

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f'Error creating evaluation sample: {e}')
            return None

    @staticmethod
    def validate_dataset(dataset: DatasetDict) -> bool:

        try:
            if 'test' not in dataset:
                print("Error: No 'test' split found")
                return False

            test_dataset = dataset['test']
            if len(test_dataset) == 0:
                print('Error: Test dataset is empty')
                return False

            required_fields = ['prompt', 'evaluation_sample', 'metadata']
            sample = test_dataset[0]

            for field in required_fields:
                if field not in sample:
                    print(f'Error: Missing required field: {field}')
                    return False

            eval_sample = sample['evaluation_sample']
            if isinstance(eval_sample, str):
                try:
                    eval_sample = json.loads(eval_sample)
                except json.JSONDecodeError:
                    print('Error: evaluation_sample is not valid JSON')
                    return False

            if not isinstance(eval_sample, dict):
                print('Error: evaluation_sample is not a dictionary')
                return False

            if 'inputs' not in eval_sample or 'outputs' not in eval_sample:
                print('Error: evaluation_sample missing inputs/outputs')
                return False

            sample_count = len(test_dataset)
            print(f'Dataset validation passed: {sample_count} samples')
            return True

        except Exception as e:
            print(f'Error during validation: {e}')
            return False
