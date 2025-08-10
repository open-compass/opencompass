import ast
import json

try:
    import networkx as nx
except ImportError:
    nx = None

import pandas as pd
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .prompts import tsp_dPrompts


def q2text(adj_matrix, distance_limit, p=tsp_dPrompts):
    total_cities = adj_matrix.shape[0]  # exclude the last row
    prompt_text = p['Intro'] + '\n' + \
                p['Initial_question'].format(total_cities=total_cities, distance_limit=distance_limit) + '\n' + \
                p['Output_content'] + '\n' + \
                p['Output_format'] + '\n' + \
                'The distances between cities are below: \n'

    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if i < j:  # only use the upper triangle
                this_line = 'The distance between City {} and City {} is {}.'.format(i, j, adj_matrix[i, j])
                prompt_text += this_line + '\n'
    return prompt_text


@LOAD_DATASET.register_module(force=True)
class CMP_TSP_D_Dataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        raw_data = []
        data_path = path
        all_data = []
        for level in range(10):
            for file_num in range(10):
                df = pd.read_csv(data_path + 'decision_data_TSP_level_{}_instance_{}.csv'.format(level, file_num + 1),
                    header=None,
                    index_col=False)
                all_data.append((level + 1, df))

        for (level, q) in all_data:
            threshold = q.iloc[-1, 0]  # therashold is the last row
            distance_matrix = q.iloc[:
                                     -1].values  # distance matrix is the rest of the rows
            prompt = q2text(distance_matrix, threshold)
            raw_data.append({
                'prompt': prompt,
                'q': str(level) + '####\n' + json.dumps(q.to_json()),
                'level': level
            })
        dataset = Dataset.from_list(raw_data)
        return dataset


@ICL_EVALUATORS.register_module(force=True)
class CMP_TSP_D_Evaluator(BaseEvaluator):

    def score(self, predictions, references):
        assert len(predictions) == len(references)

        result = {'pass': 0, 'fail': 0}
        details = {}
        tsp_d_Results = []
        for index, (q, llm_string) in enumerate(zip(references, predictions)):
            output_dict = {}
            output, reasoning = self.parse_xml_to_dict(llm_string)
            level = int(q.split('####\n')[0])
            q = json.loads(q.split('####\n')[-1])
            q = pd.DataFrame(eval(q))
            threshold = q.iloc[-1, 0]  # therashold is the last row
            distance_matrix = q.iloc[:-1].values  # distance matrix is the rest of the rows
            output_dict['output'] = output
            try:
                output_dict['correctness'], _ = self.tsp_decision_check(distance_matrix, threshold, output)
            except Exception as e:
                print(f'Check failed: {e}')
                output_dict['correctness'] = False
            output_dict['reasoning'] = reasoning
            output_dict['level'] = level
            if output_dict:
                tsp_d_Results.append(output_dict)
                if output_dict['correctness']:
                    r = 'pass'
                else:
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

    def tsp_approx(self, distance_matrix):
        """Returns an approximate solution to the TSP problem.

        :param distance_matrix: A 2D numpy array representing the distance matrix.
        :return: A list of the cities in the order they were visited.
        """
        G = nx.from_numpy_array(distance_matrix)
        return nx.approximation.traveling_salesman_problem(G)

    def tsp_decision_check(self, distance_matrix, threshold, tour):
        """Checks if a given TSP tour is valid and within the threshold
        distance.

        :param distance_matrix: A 2D numpy array representing the distance matrix.
        :param threshold: The maximum distance allowed.
        :param tour: A dictionary containing the feasibility.
        """
        try:
            is_feasible = tour.get('Feasible', 'no').lower() == 'yes'
        except Exception:
            return False, 'Output format incorrect'

        # Calculate the approxed distance of the tour
        tours = self.tsp_approx(distance_matrix)
        tour_distance = sum(distance_matrix[tours[i], tours[i + 1]] for i in range(len(tours) - 1)) + distance_matrix[tours[-1], tours[0]]

        if is_feasible != (tour_distance <= threshold):
            return False, f'Feasibility mismatch: {is_feasible} vs {tour_distance} > {threshold}'
        return True, 'Feasible: {} <= {}'.format(tour_distance, threshold)
