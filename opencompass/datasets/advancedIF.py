import json
import re

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET
from opencompass.utils import get_logger

from .base import BaseDataset

logger = get_logger()


@LOAD_DATASET.register_module()
class AdvancedIFDataset(BaseDataset):

    @staticmethod
    def load(path, **kwargs):
        raw_dataset = load_dataset(path)

        dataset = []
        for item in raw_dataset['train']:
            # Parse conversation_history from JSON string
            conversation_history = json.loads(item['conversation_history'])

            # Extract the last user turn as a separate column
            last_user_question = ''
            last_user_idx = -1
            for i in range(len(conversation_history) - 1, -1, -1):
                if conversation_history[i].get('role') == 'user':
                    last_user_question = conversation_history[i].get(
                        'content', '')
                    last_user_idx = i
                    break

            # full_conversation: conversation_history with the last user
            # turn removed, serialized as JSON string
            if last_user_idx >= 0:
                truncated_history = (conversation_history[:last_user_idx]
                                     + conversation_history[last_user_idx + 1:])
            else:
                truncated_history = conversation_history
            full_conversation = json.dumps(truncated_history,
                                           ensure_ascii=False)

            # Extract rubrics from prompt_metadata
            # prompt_metadata is a JSON string like {"rubrics": "[...]"}
            # where the rubrics field itself is a string containing the
            # rubric questions
            prompt_metadata = item.get('prompt_metadata', '')
            rubrics_text = ''
            if isinstance(prompt_metadata, str) and prompt_metadata:
                try:
                    metadata = json.loads(prompt_metadata)
                    rubrics_text = metadata.get('rubrics', '')
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(prompt_metadata, dict):
                rubrics_text = prompt_metadata.get('rubrics', '')

            dataset.append({
                'conversation_history': conversation_history,
                'full_conversation': full_conversation,
                'last_user_question': last_user_question,
                'rubrics_text': rubrics_text,
            })

        test_dataset = Dataset.from_list(dataset)
        return test_dataset


def _extract_rubric_judgement(prediction: str) -> dict:
    """Extract full judgement from judge model output.

    Searches for a JSON blob containing rubrics_check and
    SATISFIED_ALL_REQUIREMENTS.

    Returns:
        dict with keys:
            - satisfied: 'YES', 'NO', or 'PARSE_ERROR'
            - rubrics_check: dict of {question_key: answer} or {}
            - raw_json: the parsed JSON dict or None
    """
    result = {
        'satisfied': 'PARSE_ERROR',
        'rubrics_check': {},
        'raw_json': None,
    }

    # Try to find JSON blob in the output
    json_match = re.search(r'\{[\s\S]*?"SATISFIED_ALL_REQUIREMENTS"[\s\S]*\}',
                           prediction, re.DOTALL)
    if not json_match:
        json_match = re.search(
            r'\{[^{}]*"SATISFIED_ALL_REQUIREMENTS"[^{}]*\}',
            prediction, re.DOTALL)

    if json_match:
        try:
            parsed = json.loads(json_match.group())
            result['raw_json'] = parsed
            result['rubrics_check'] = parsed.get('rubrics_check', {})

            satisfied = parsed.get('SATISFIED_ALL_REQUIREMENTS', '')
            if isinstance(satisfied, str):
                satisfied_upper = satisfied.strip().upper()
                if satisfied_upper in ('YES', 'NO'):
                    result['satisfied'] = satisfied_upper
                    return result
        except json.JSONDecodeError:
            pass

    # Fallback: look for YES/NO near SATISFIED_ALL_REQUIREMENTS
    satisfied_match = re.search(
        r'"SATISFIED_ALL_REQUIREMENTS"\s*:\s*"(YES|NO)"',
        prediction, re.IGNORECASE)
    if satisfied_match:
        result['satisfied'] = satisfied_match.group(1).upper()

    return result


@DICT_POSTPROCESSORS.register_module()
def advancedif_rubric_postprocess(output: dict, output_path: str) -> dict:
    """Postprocess the rubric judge output.

    Parses the JSON blob from each judge prediction, extracts
    SATISFIED_ALL_REQUIREMENTS and rubrics_check, then computes:
    - accuracy: sample-level pass rate (SATISFIED_ALL_REQUIREMENTS == YES)
    - micro_pass_rate: rubric-question-level pass rate (total passed
      questions / total questions across all samples)
    """
    details = []
    correct_count = 0
    parse_error_count = 0
    total_rubrics = 0
    passed_rubrics = 0
    total = 0

    for k, v in output.items():
        prediction = v.get('prediction', '')
        gold = v.get('gold', '')
        judgement = _extract_rubric_judgement(prediction)
        satisfied = judgement['satisfied']
        rubrics_check = judgement['rubrics_check']

        total += 1

        # Sample-level: SATISFIED_ALL_REQUIREMENTS
        if satisfied == 'YES':
            correct_count += 1
        elif satisfied == 'PARSE_ERROR':
            parse_error_count += 1

        # Rubric-question-level: count individual rubric passes
        for question_key, decision_value in rubrics_check.items():
            total_rubrics += 1
            if isinstance(decision_value, str) and 'yes' in decision_value.lower():
                passed_rubrics += 1

        details.append({
            'prediction': prediction,
            'gold': gold,
            'satisfied': satisfied,
            'rubrics_check': rubrics_check,
            'correct': satisfied == 'YES',
        })

    accuracy = (correct_count / total * 100) if total > 0 else 0
    micro_pass_rate = (passed_rubrics / total_rubrics *
                       100) if total_rubrics > 0 else 0
    parse_error_rate = (parse_error_count / total *
                        100) if total > 0 else 0

    result = {
        'accuracy': accuracy,
        'micro_pass_rate': micro_pass_rate,
        'correct_count': correct_count,
        'total': total,
        'total_rubrics': total_rubrics,
        'passed_rubrics': passed_rubrics,
        'parse_error_count': parse_error_count,
        'parse_error_rate': parse_error_rate,
        'details': details,
    }
    return result
