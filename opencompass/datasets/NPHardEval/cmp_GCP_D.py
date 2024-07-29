import ast

try:
    import networkx as nx
except ImportError:
    nx = None

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .prompts import gcp_dPrompts


def q2text(q, p=gcp_dPrompts):
    number_of_colors = q.split('\n')[0].split()[-2]  # last character of the first line
    number_of_vertices = q.split('\n')[1].split(' ')[2]  # third word of the second line
    prompt_text =   p['Intro'] + '\n' + \
                    p['Initial_question'].format(total_vertices=number_of_vertices, number_of_colors=number_of_colors) + '\n' + \
                    p['Output_content'] + '\n' + \
                    p['Output_format'] + '\n' + \
                    '\n The graph is below: \n'
    for line in q.split('\n')[2:]:
        vertex_list = line.split(' ')
        this_line = 'Vertex {} is connected to vertex {}.'.format(
            vertex_list[1], vertex_list[2])
        prompt_text += this_line + '\n'
    return prompt_text


@LOAD_DATASET.register_module(force=True)
class CMP_GCP_D_Dataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        raw_data = []
        data_path = path
        all_data = []
        for file_num in range(10):
            with open(data_path + 'decision_data_GCP_{}.txt'.format(file_num)) as f:
                data = f.read()
                sample = data.split('\n\n')[:-1]
            all_data += zip([file_num + 1] * len(sample), sample)
        for (level, q) in all_data:
            prompt = q2text(q)
            raw_data.append({
                'prompt': prompt,
                'q': str(level) + '####\n' + q,
                'level': level
            })
        dataset = Dataset.from_list(raw_data)
        return dataset


@ICL_EVALUATORS.register_module(force=True)
class CMP_GCP_D_Evaluator(BaseEvaluator):

    def score(self, predictions, references):
        assert len(predictions) == len(references)

        result = {'pass': 0, 'fail': 0}
        details = {}
        for index, (q, output) in enumerate(zip(references, predictions)):
            output_dict = {}
            level = int(q.split('####\n')[0])
            q = q.split('####\n')[-1]
            try:
                number_of_colors = int(q.split('\n')[0].split()[-2])
                output, reasoning = self.parse_xml_to_dict(output)
                output_dict['output'] = output
                output_dict['correctness'], _ = self.gcp_decision_check(q, output, number_of_colors)
            except Exception as e:
                print(f'Attempt failed: {e}')
                output_dict['correctness'] = False
            output_dict['reasoning'] = reasoning

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

    def read_dimacs_format(self, dimacs_str):
        lines = dimacs_str.strip().split('\n')
        p_line = next(line for line in lines if line.startswith('p'))
        _, _, num_vertices, num_edges = p_line.split()
        num_vertices, num_edges = int(num_vertices), int(num_edges)

        adjacency_list = {i: set() for i in range(1, num_vertices + 1)}
        for line in lines:
            if line.startswith('e'):
                _, vertex1, vertex2 = line.split()
                vertex1, vertex2 = int(vertex1), int(vertex2)
                if vertex1 in adjacency_list and vertex2 in adjacency_list:
                    adjacency_list[vertex1].add(vertex2)
                    adjacency_list[vertex2].add(vertex1)

        return num_vertices, adjacency_list

    def gcp_greedy_solution(self, adjacency_list):
        """Provides a greedy solution to the GCP problem.

        :param adjacency_list: A dictionary of the adjacency list.
        :return: A tuple of (num_colors, coloring).
        """
        G = nx.Graph()
        G.add_nodes_from(adjacency_list.keys())
        for vertex, neighbors in adjacency_list.items():
            for neighbor in neighbors:
                G.add_edge(vertex, neighbor)
        coloring = nx.coloring.greedy_color(G, strategy='largest_first')
        num_colors = max(coloring.values()) + 1
        return num_colors, coloring

    def gcp_decision_check(self, dimacs_str, answer, k_colors):
        """Check if the given GCP instance is feasible with k_colors.

        :param dimacs_str: The DIMACS format string of the GCP instance.
        :param answer: The answer returned by the model.
        :param k_colors: The target number of colors.
        :return: A tuple of (is_correct, message).
        """
        num_vertices, adjacency_list = self.read_dimacs_format(dimacs_str)
        try:
            is_feasible = answer.get('Feasible', 'no').lower() == 'yes'
        except Exception:
            return False, 'Feasible key not found'
        num_colors, coloring = self.gcp_greedy_solution(adjacency_list)
        exist_optimal = num_colors <= k_colors
        if is_feasible != exist_optimal:
            if exist_optimal:
                return False, f'Feasibility mismatch: {coloring}'
            else:
                return False, f'Feasibility mismatch: {is_feasible} vs {exist_optimal}'
        return True, 'Feasible' if is_feasible else 'Infeasible'
