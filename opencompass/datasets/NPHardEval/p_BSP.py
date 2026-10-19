import ast
import json

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .prompts import bspPrompts


def q2text(q, p=bspPrompts):
    target_value = q['target']
    # TO-DO: fix data not being sorted
    array = sorted(q['array'])
    prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(target_value=target_value) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format'] + \
                  '\n The sorted array elements are: ' + ', '.join(map(str, array)) + '\n'

    return prompt_text


@LOAD_DATASET.register_module(force=True)
class P_BSP_Dataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        raw_data = []
        data_path = path
        all_data, newdata = [], []
        with open(data_path + 'bsp_instances.json', 'r') as f:
            data = json.load(f)
            for sample in data:
                level = len(sample['array']) - 2
                all_data.append((level, sample))

        for level, q in all_data:
            prompt = q2text(q)
            raw_data.append({
                'prompt': prompt,
                'q': str(level) + '####\n' + json.dumps(q),
                'level': level
            })
        dataset = Dataset.from_list(raw_data)
        return dataset


@ICL_EVALUATORS.register_module(force=True)
class P_BSP_Evaluator(BaseEvaluator):

    def score(self, predictions, references):
        assert len(predictions) == len(references)

        result = {'pass': 0, 'fail': 0}
        for index, (q, output) in enumerate(zip(references, predictions)):
            output_dict = {}
            level = int(q.split('####\n')[0])
            q = json.loads(q.split('####\n')[-1])
            output, reasoning = self.parse_xml_to_dict(output)
            output_dict['output'] = output
            try:
                output_dict['correctness'], _ = self.bsp_check(q, output)
            except Exception as e:
                print(f'Check failed: {e}')
                output_dict['correctness'] = False
            output_dict['reasoning'] = reasoning
            output_dict['level'] = level

            if output_dict['correctness']:
                r = 'pass'
            else:
                r = 'fail'
            result[r] += level

        result['score'] = result['pass'] / (result['pass'] + result['fail']) * 100
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

    def bsp_check(self, instance, solution):
        """Check if the binary search solution is valid.

        :param instance: The instance dictionary with array and target value.
        :param solution: The solution dictionary with the position of the target value.
        :return: A tuple of (is_correct, message).
        """
        array = sorted(instance['array'])
        target_value = instance['target']
        solution, reasoning = self.parse_xml_to_dict(solution)
        if isinstance(solution, str):
            return False, f'The solution is invalid.'
        try:
            position = int(solution['Position'])
        except Exception:
            return False, f'The solution is invalid.'
        if position == -1 or position >= len(array):
            return False, f'The solution is invalid.'
        elif array[position] != target_value:
            return False, f'The target index is incorrect.'
        return True, 'The solution is valid.'
