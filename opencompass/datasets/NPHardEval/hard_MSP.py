import ast
import json
import xml.etree.ElementTree as ET

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .prompts import mspPrompts


def q2text(q, p=mspPrompts):  # q is the data for the HP-hard question, p is the prompt
    total_participants = q['participants']
    total_timeslots = q['time_slots']
    prompt_text = p['Intro'] + '\n' \
        + p['Initial_question'].format(total_participants=total_participants,total_timeslots=total_timeslots) + '\n' \
        + p['Output_content'] + '\n' \
        + p['Output_format'] + \
        '\n The meetings and participants details are as below: \n'
    meetings = q['meetings']
    participants = q['participants']
    for meeting in meetings:
        this_line = 'Meeting {} is with duration {}.'.format(meeting['id'], meeting['duration'])
        prompt_text += this_line + '\n'
    for j in participants.keys():
        this_line = 'Participant {} is available at time slots {} and has meetings {}.'.format(j, participants[j]['available_slots'], participants[j]['meetings'])
        prompt_text += this_line + '\n'
    return prompt_text


@LOAD_DATASET.register_module(force=True)
class Hard_MSP_Dataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        raw_data = []
        data_path = path
        all_data = []
        with open(data_path + 'msp_instances.json', 'r') as f:
            data = json.load(f)
            all_data = zip([int(d['complexity_level']) for d in data], data)

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
class Hard_MSP_Evaluator(BaseEvaluator):

    def score(self, predictions, references):
        assert len(predictions) == len(references)

        result = {'pass': 0, 'fail': 0}
        for index, (q, output) in enumerate(zip(references, predictions)):
            output_dict = {}
            level = int(q.split('####\n')[0])
            q = json.loads(q.split('####\n')[-1])

            output_dict['output'] = output
            output_dict['level'] = level
            try:
                output_dict['correctness'], _ = self.mspCheck(q, output)
            except Exception as e:
                print(f'Check failed: {e}')
                output_dict['correctness'] = False

            if output_dict['correctness']:
                r = 'pass'
            else:
                r = 'fail'
            result[r] += level

        result['score'] = result['pass'] / (result['pass'] + result['fail']) * 100
        final_result = {'Weighted Accuracy': result['score']}
        return final_result

    def mspCheck(self, instance, llm_string):
        """Validate the MSP solution.

        Parameters:
        - instance: The MSP instance as a dictionary.
        - solution: A dictionary with meeting ids as keys and lists of scheduled time slots as values.

        Returns:
        - A tuple (is_valid, message). is_valid is True if the solution is valid, False otherwise.
        message contains information about the validity of the solution.
        """
        # print(llm_string)
        solution, reasoning_element = self.parse_xml_to_dict(llm_string)
        # print(solution.text)

        # convert solution to dictionary
        if solution == '':
            return False, None
        elif solution is None:
            return False, None
        else:
            if isinstance(solution, str):
                try:
                    solution = ast.literal_eval(solution)
                    if solution is None:
                        return False, None
                except Exception:
                    try:
                        solution = ast.literal_eval('{' + solution + '}')
                        if solution is None:
                            return False, None
                    except Exception:
                        return False, None
            else:
                try:
                    solution = ast.literal_eval(solution.text)
                    if solution is None:
                        return False, None
                except Exception:
                    return False, None
        # convert key type to int
        if isinstance(solution, dict):
            print(solution)
            solution = {int(k): v for k, v in solution.items()}
        else:
            return False, None

        # Check if all meetings are scheduled within the available time slots
        for meeting in instance['meetings']:
            m_id = meeting['id']
            duration = meeting['duration']
            scheduled_slots = solution.get(m_id, None)

            # Check if the meeting is scheduled
            if scheduled_slots is None:
                return False, f'Meeting {m_id} is not scheduled.'

            # Check if the meeting fits within the number of total time slots
            if any(slot >= instance['time_slots'] for slot in scheduled_slots):
                return False, f'Meeting {m_id} does not fit within the available time slots.'

            # Check if the scheduled slots are contiguous and fit the meeting duration
            if len(scheduled_slots) != duration or not all(scheduled_slots[i] + 1 == scheduled_slots[i + 1]
                    for i in range(len(scheduled_slots) - 1)):
                return False, f'Meeting {m_id} is not scheduled in contiguous time slots fitting its duration.'

            # Check if all participants are available at the scheduled time
            for p_id, participant in instance['participants'].items():
                if m_id in participant['meetings']:
                    if not all(slot in participant['available_slots'] for slot in scheduled_slots):
                        return False, f'Participant {p_id} is not available for meeting {m_id} at the scheduled time.'

        # Check if any participant is double-booked
        participants_schedule = {p_id: [] for p_id in instance['participants']}
        for m_id, time_slots in solution.items():
            try:
                duration = next(meeting['duration'] for meeting in instance['meetings'] if meeting['id'] == m_id)
                if len(time_slots) != duration:
                    return False, f'Meeting {m_id} duration does not match the number of scheduled time slots.'
                for p_id, participant in instance['participants'].items():
                    if m_id in participant['meetings']:
                        participants_schedule[p_id].extend(time_slots)
            except Exception:
                return False, f'Meeting {m_id} is not in the instance or program error.'

        for p_id, slots in participants_schedule.items():
            if len(slots) != len(set(slots)):
                return False, f'Participant {p_id} is double-booked.'

        return True, 'The solution is valid.'

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
