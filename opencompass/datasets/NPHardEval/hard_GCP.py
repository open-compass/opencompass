import ast
import xml.etree.ElementTree as ET

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .prompts import gcpPrompts


def q2text(q, p=gcpPrompts):  # q is the data for the HP-hard question, p is the prompt
    # print(q)
    chromatic_number = q.split('\n')[0][-1]  # last character of the first line
    number_of_vertices = q.split('\n')[1].split(' ')[2]  # third word of the second line
    prompt_text = p['Intro'] + '\n' \
        + p['Initial_question'].format(max_vertices=number_of_vertices,max_colors=chromatic_number) + '\n' \
        + p['Output_content'] + '\n' \
        + p['Output_format'] + \
        '\n The graph is below: \n'
    for line in q.split('\n')[2:]:
        vertex_list = line.split(' ')
        this_line = 'Vertex {} is connected to vertex {}.'.format(vertex_list[1], vertex_list[2])
        prompt_text += this_line + '\n'

    return prompt_text


@LOAD_DATASET.register_module(force=True)
class HardGCPDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        raw_data = []
        data_path = path
        all_data = []
        for file_num in range(10):
            with open(data_path + 'synthesized_data_GCP_{}.txt'.format(file_num)) as f:
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
class HardGCPEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        assert len(predictions) == len(references)

        result = {'pass': 0, 'fail': 0}
        details = {}
        for index, (q, output) in enumerate(zip(references, predictions)):
            output_dict = {}
            level = int(q.split('####\n')[0])
            q = q.split('####\n')[-1]

            output_dict['output'] = output
            try:
                output_dict['correctness'] = self.gcpCheck(q, output)
            except Exception as e:
                print(f'Check failed: {e}')
                output_dict['correctness'] = False
            output_dict['level'] = level

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
            # Parse the XML string
            root = ET.fromstring(xml_string)

            # Find the 'final_answer' tag
            final_answer_element = root.find('final_answer')

            # Find the 'reasoning' tag
            reasoning_element = root.find('reasoning')
        except Exception:
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
            except Exception:
                final_answer_element = ''
                reasoning_element = ''

        return final_answer_element, reasoning_element

    def gcpCheck(self, dimacs_str, answer_str):
        num_vertices, adjacency_list = self.read_dimacs_format(dimacs_str)
        answer_colors = self.parse_answer(answer_str)
        # print(adjacency_list)
        # print(answer_colors)

        # Check if all colors in the answer are valid
        for vertex, neighbors in adjacency_list.items():
            for neighbor in neighbors:
                try:
                    if answer_colors[vertex] == answer_colors[neighbor]:
                        print(f'Invalid coloring: Vertex {vertex} and {neighbor} have the same color.')
                        return False
                except:
                    print(f'Invalid input.')  # dealing with hullucination
                    return False

        print(f'Valid coloring found with {len(set(answer_colors.values()))} colors: {answer_colors}')
        return True

    def read_dimacs_format(self, dimacs_str):
        lines = dimacs_str.strip().split('\n')
        # Read the number of vertices and edges
        p_line = next(line for line in lines if line.startswith('p'))
        _, _, num_vertices, num_edges = p_line.split()
        num_vertices, num_edges = int(num_vertices), int(num_edges)

        # Create adjacency list
        adjacency_list = {i: set() for i in range(1, num_vertices + 1)}

        # Read the edges and ignore those that reference non-existing vertices
        for line in lines:
            if line.startswith('e'):
                _, vertex1, vertex2 = line.split()
                vertex1, vertex2 = int(vertex1), int(vertex2)
                if vertex1 in adjacency_list and vertex2 in adjacency_list:
                    adjacency_list[vertex1].add(vertex2)
                    adjacency_list[vertex2].add(vertex1)

        return num_vertices, adjacency_list

    def parse_answer(self, llm_string):
        # # Convert the answer string to a dictionary
        # answer_dict = {}
        # # Remove the braces and split the string by commas
        # entries = answer_str.strip("}{").split(', ')
        # for entry in entries:
        #     vertex, color = entry.split(':')
        #     answer_dict[int(vertex)] = color
        # return answer_dict

        all_answers, reasoning_element = self.parse_xml_to_dict(llm_string)

        if all_answers == '':
            return {}
        elif all_answers is None:
            return {}
        else:
            if isinstance(all_answers, str):
                try:
                    all_answers = ast.literal_eval(all_answers)
                except Exception:
                    try:
                        all_answers = ast.literal_eval('{' + all_answers + '}')
                    except Exception:
                        return {}
            else:
                all_answers = ast.literal_eval(all_answers.text)
        # answer_dict = {}
        # for pair in all_answers:
        #     vertex, color = pair.split(":")
        #     answer_dict[int(vertex)] = color
        # convert key type to int
        all_answers = {int(k): v for k, v in all_answers.items()}
        return all_answers  # answer_dict
