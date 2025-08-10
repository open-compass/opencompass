import ast
import json

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .prompts import edpPrompts


def q2text(q, p=edpPrompts):
    string_a = q['string_a']
    string_b = q['string_b']
    prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(string_a=string_a, string_b=string_b) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format']
    return prompt_text


@LOAD_DATASET.register_module(force=True)
class P_EDP_Dataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        raw_data = []
        data_path = path
        all_data = []
        with open(data_path + 'edp_instances.json', 'r') as f:
            data = json.load(f)
            for sample in data:
                level = len(sample['string_a']) - 2
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
class P_EDP_Evaluator(BaseEvaluator):

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
                output_dict['correctness'], _ = self.edp_check(q, output)
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

    def compute_min_edit_distance(self, string_a, string_b):
        """Computes the minimum edit distance between two strings using dynamic
        programming."""
        m, n = len(string_a), len(string_b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif string_a[i - 1] == string_b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    def edp_check(self, instance, solution):
        """Check if the edit distance solution is valid.

        :param instance: The instance dictionary with 'string_a' and 'string_b'.
        :param solution: The solution dictionary with the reported 'edit_distance'.
        :return: A tuple of (is_correct, message).
        """
        string_a = instance['string_a']
        string_b = instance['string_b']
        try:
            reported_distance = int(solution.get('Operations', -1))
        except Exception:
            reported_distance = -1

        actual_distance = self.compute_min_edit_distance(string_a, string_b)

        if reported_distance == -1:
            return False, 'No solution provided.'
        elif reported_distance != actual_distance:
            return False, f'The reported edit distance ({reported_distance}) is incorrect. Actual distance: {actual_distance}.'
        return True, 'The solution is valid.'

    def parse_xml_to_dict(self, xml_string):
        try:
            assert '<final_answer>' in xml_string
            assert '</final_answer>' in xml_string
            # assert '<reasoning>' in xml_string
            # assert '</reasoning>' in xml_string
            final_answer_start = xml_string.index('<final_answer>') + len('<final_answer>')
            final_answer_end = xml_string.index('</final_answer>')
            # reasoning_start = xml_string.index('<reasoning>') + len('<reasoning>')
            # reasoning_end = xml_string.index('</reasoning>')
            final_answer_element = xml_string[final_answer_start:final_answer_end].rstrip().strip().rstrip()
            assert '{' in final_answer_element
            assert '}' in final_answer_element
            dic_start = final_answer_element.index('{')
            dic_end = final_answer_element.index('}')
            final_answer_element = final_answer_element[dic_start:dic_end + 1].rstrip().strip().rstrip()
            reasoning_element = xml_string
            # reasoning_element = xml_string[reasoning_start:reasoning_end].rstrip().strip().rstrip()
            try:
                final_answer_element = ast.literal_eval(final_answer_element)
            except Exception:
                final_answer_element = ''
        except Exception:
            final_answer_element = ''
            reasoning_element = ''

        return final_answer_element, reasoning_element
