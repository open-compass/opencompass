import ast
import json

try:
    import networkx as nx
except ImportError:
    nx = None

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .prompts import sppPrompts


def q2text(q, p=sppPrompts):
    # start_node = q['start_node']
    # end_node = q['end_node']
    # TO-DO: fix later
    start_node = q['nodes'][0]
    end_node = q['nodes'][-1]
    edges = q['edges']
    prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(start_node=start_node, end_node=end_node) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format'] + \
                  "\n The graph's edges and weights are as follows: \n"
    for edge in edges:
        this_line = f"Edge from {edge['from']} to {edge['to']} has a weight of {edge['weight']}."
        prompt_text += this_line + '\n'
    return prompt_text


@LOAD_DATASET.register_module(force=True)
class P_SPP_Dataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        raw_data = []
        data_path = path
        all_data = []
        with open(data_path + 'spp_instances.json', 'r') as f:
            data = json.load(f)
            all_data = zip([int(d['complexity_level']) for d in data], data)
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
class P_SPP_Evaluator(BaseEvaluator):

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
                output_dict['correctness'], _ = self.spp_check(q, output)
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
            # reasoning_element = xml_string[reasoning_start:reasoning_end].rstrip().strip().rstrip()
            try:
                final_answer_element = ast.literal_eval(final_answer_element)
                reasoning_element = xml_string
            except Exception:
                final_answer_element = ''
                reasoning_element = xml_string
        except Exception:
            final_answer_element = ''
            reasoning_element = ''

        return final_answer_element, reasoning_element

    def ssp_optimal_solution(self, instance, source, target):
        """Provides the optimal solution for the SSP instance.

        :param instance: The SSP instance as a dictionary with 'nodes' and 'edges'.
        :param source: The source node.
        :param target: The destination node.
        :return: The optimal shortest path length and path.
        """
        G = nx.Graph()
        G.add_nodes_from(instance['nodes'])
        G.add_weighted_edges_from([(edge['from'], edge['to'], edge['weight'])
                                   for edge in instance['edges']])
        shortest_path_length = None
        shortest_path = None
        if nx.has_path(G, source=source, target=target):
            shortest_path_length = nx.shortest_path_length(G, source=source, target=target, weight='weight')
            shortest_path = nx.shortest_path(G, source=source, target=target, weight='weight')
        return shortest_path_length, shortest_path

    # SPP
    def spp_check(self, instance, solution, start_node=None, end_node=None):
        """Validate the solution of the SPP problem.

        :param instance: The instance dictionary with nodes and edges.
        :param solution: The solution dictionary with the path and total distance.
        :param start_node: The start node.
        :param end_node: The end node.
        :return: A tuple of (is_correct, message).
        """
        # Get the start and end nodes
        # Currently, the start and end nodes are the first and last nodes in the instance
        if start_node is None:
            start_node = instance['nodes'][0]
        if end_node is None:
            end_node = instance['nodes'][-1]

        # Convert solution to dictionary
        try:
            path_string = solution.get('Path', '')
            cost_string = solution.get('TotalDistance', '')
        except Exception:
            return False, 'The solution is not a dictionary.'

        # Calculate the optimal solution
        ssp_optimal_length, ssp_optimal_path = self.ssp_optimal_solution(
            instance, start_node, end_node)
        if ssp_optimal_length is None:
            if isinstance(cost_string, int) or cost_string.isdigit():
                return False, f'No path between from node {start_node} to node {end_node}.'
            else:
                return True, 'No path found from node {start_node} to node {end_node}.'

        try:
            path = list(map(int, path_string.split('->')))
            total_cost = int(cost_string)
        except Exception:
            return False, 'The solution is not a valid dictionary.'

        # Check if path starts and ends with the correct nodes
        if not path or path[0] != start_node or path[-1] != end_node:
            return False, 'The path does not start or end at the correct nodes.'

        # Check if the path is continuous and calculate the cost
        calculated_cost = 0
        is_in_edge = lambda edge, from_node, to_node: (edge['from'] == from_node and edge['to'] == to_node) or (edge['from'] == to_node and edge['to'] == from_node)
        for i in range(len(path) - 1):
            from_node, to_node = path[i], path[i + 1]
            edge = next((edge for edge in instance['edges'] if is_in_edge(edge, from_node, to_node)), None)

            if not edge:
                return False, f'No edge found from node {from_node} to node {to_node}.'

            calculated_cost += edge['weight']

        # Check if the calculated cost matches the total cost provided in the solution
        if calculated_cost != total_cost:
            return False, f'The calculated cost ({calculated_cost}) does not match the provided total cost ({total_cost}).'

        if calculated_cost != ssp_optimal_length:
            # spp_optimal_path = '->'.join(map(str, ssp_optimal_path))
            return False, f'The calculated cost ({calculated_cost}) does not match the optimal solution ({ssp_optimal_length}): {ssp_optimal_path}.'

        return True, 'The solution is valid.'
