import ast
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .prompts import tspPrompts


def q2text(q, p=tspPrompts):  # q is the data for the HP-hard question, p is the prompt
    total_cities = q.shape[0]
    prompt_text = p['Intro'] + '\n' \
        + p['Initial_question'].format(total_cities=total_cities) + '\n' \
        + p['Output_content'] + '\n' \
        + p['Output_format'] + \
        '\n The distances between cities are below: \n'
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            if i < j:  # only use the upper triangle
                this_line = 'The path between City {} and City {} is with distance {}.'.format(i, j, q.iloc[i, j])
                prompt_text += this_line + '\n'
    return prompt_text


@LOAD_DATASET.register_module(force=True)
class Hard_TSP_Dataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        raw_data = []
        data_path = path
        all_data = []
        for level in range(10):
            for file_num in range(10):
                # read np array
                df = pd.read_csv(data_path + 'synthesized_data_TSP_level_{}_instance_{}.csv'.format(level, file_num + 1),
                    header=None,
                    index_col=False)
                # transform df to
                all_data.append((level + 1, df))
        for (level, q) in all_data:
            prompt = q2text(q)
            raw_data.append({
                'prompt': prompt,
                'q': str(level) + '####\n' + json.dumps(q.to_json()),
                'level': level
            })
        dataset = Dataset.from_list(raw_data)
        return dataset


@ICL_EVALUATORS.register_module(force=True)
class Hard_TSP_Evaluator(BaseEvaluator):

    def score(self, predictions, references):
        assert len(predictions) == len(references)

        result = {'pass': 0, 'fail': 0}
        for index, (q, output) in enumerate(zip(references, predictions)):
            output_dict = {}
            level = int(q.split('####\n')[0])
            q = json.loads(q.split('####\n')[-1])
            q = pd.DataFrame(eval(q))

            output_dict['output'] = output
            try:
                output_dict['correctness'], _ = self.tspCheck(q, output)
            except Exception as e:
                print(f'Check failed: {e}')
                output_dict['correctness'] = False
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
            # Parse the XML string
            root = ET.fromstring(xml_string)

            # Find the 'final_answer' tag
            final_answer_element = root.find('final_answer')

            # Find the 'reasoning' tag
            reasoning_element = root.find('reasoning')
        except:
            try:
                assert '<final_answer>' in xml_string
                assert '</final_answer>' in xml_string
                assert '<reasoning>' in xml_string
                assert '</reasoning>' in xml_string
                final_answer_start = xml_string.index('<final_answer>') + len('<final_answer>')
                final_answer_end = xml_string.index('</final_answer>')
                reasoning_start = xml_string.index('<reasoning>') + len('<reasoning>')
                reasoning_end = xml_string.index('</reasoning>')
                final_answer_element = xml_string[final_answer_start:final_answer_end]
                reasoning_element = xml_string[reasoning_start:reasoning_end]
            except:
                final_answer_element = ''
                reasoning_element = ''

        return final_answer_element, reasoning_element

    def tspCheck(self, distance_matrix, llm_string):
        """Check if the TSP solution is complete and if the distance matches
        the greedy solution.

        :param tour_string: String representing the TSP tour in the format "0->1->2->...->N->0"
        :param distance_matrix: 2D numpy array representing the distances between cities
        :return: Boolean indicating whether the tour is complete and matches the greedy distance
        """
        # convert distance_matrix to numpy array
        distance_matrix = np.array(distance_matrix)

        # Convert the tour string to a list of integers
        # print(llm_string)
        final_answer_element, reasoning_element = self.parse_xml_to_dict(llm_string)
        # convert solution to dictionary
        if final_answer_element == '':
            return False, ''
        elif final_answer_element is None:
            return False, ''
        else:
            if isinstance(final_answer_element, str):
                try:
                    tour_string = ast.literal_eval(final_answer_element)['Path']
                    if tour_string is None:
                        return False, ''
                except Exception:
                    try:
                        tour_string = ast.literal_eval('{' + final_answer_element + '}')['Path']
                        if tour_string is None:
                            return False, ''
                    except Exception:
                        return False, ''
            else:
                try:
                    tour_string = ast.literal_eval(final_answer_element.text)['Path']
                    if tour_string is None:
                        return False, ''
                except Exception:
                    return False, ''
        try:
            tour = list(map(int, tour_string.split('->')))
        except Exception:
            return False, ''
        # we could also prinpt `reasoning_element` to see the reasoning of the answer
        # we could also print the final distance of the tour by `final_answer_element['Distance']`

        # Check if tour is a cycle
        if tour[0] != tour[-1]:
            return False, 'The tour must start and end at the same city.'

        # Check if all cities are visited
        if len(tour) != len(distance_matrix) + 1:
            return False, 'The tour does not visit all cities exactly once.'

        # Calculate the distance of the provided tour
        tour_distance = sum(distance_matrix[tour[i]][tour[i + 1]]
                            for i in range(len(tour) - 1))

        # Find the greedy tour distance for comparison
        greedy_tour, greedy_distance = self.greedy_tsp(distance_matrix)

        # Check if the provided tour distance is equal to the greedy tour distance
        if tour_distance != greedy_distance:
            return False, f'The tour distance ({tour_distance}) does not match the greedy solution ({greedy_distance}).'

        return True, 'The solution is complete and matches the greedy solution distance.'

    def greedy_tsp(self, distance_matrix):
        """Solve the Traveling Salesman Problem using a greedy algorithm.

        :param distance_matrix: 2D numpy array where the element at [i, j] is the distance between city i and j
        :return: A tuple containing a list of the cities in the order they were visited and the total distance
        """
        num_cities = distance_matrix.shape[0]
        unvisited_cities = set(range(num_cities))
        current_city = np.random.choice(list(unvisited_cities))
        tour = [current_city]
        total_distance = 0

        while unvisited_cities:
            unvisited_cities.remove(current_city)
            if unvisited_cities:
                # Find the nearest unvisited city
                distances_to_unvisited = distance_matrix[current_city][list(unvisited_cities)]
                nearest_city = list(unvisited_cities)[np.argmin(distances_to_unvisited)]
                tour.append(nearest_city)
                # Update the total distance
                total_distance += distance_matrix[current_city, nearest_city]
                current_city = nearest_city

        # Return to start
        total_distance += distance_matrix[current_city, tour[0]]
        tour.append(tour[0])

        return tour, total_distance
