import re

from opencompass.registry import DICT_POSTPROCESSORS
from opencompass.utils import get_logger

# Anchored grade marker: a grade cue word immediately followed by a connector
# and the grade letter. Cues are deliberately narrow (grade/verdict/final
# answer) — bare "answer" is excluded because "the answer is A" names the
# *option*, not the grade; only "final answer" is unambiguous. A connector is
# required so bare prose like "grade A of this study" (a quality qualifier)
# cannot be mistaken for a grade. Connector grammar covers both word forms
# ("grade is B", "grade of A", "final answer is: B") and symbol forms
# ("grade: B", "grade = B", "grade:=B").
_ANCHORED_GRADE_RE = re.compile(
    r'\b(?:grade|verdict|final\s+answer)\s*'
    r'(?:(?:is|was|be|of)\s*[:=]?|[:=]{1,2})'
    r'\s*[(\[]?\s*([AB])\s*[)\]]?',
    re.IGNORECASE)


def get_final_results(judged_answers,
                      references,
                      origial_responses,
                      metric_name='accuracy',
                      true_tag: str = 'A',
                      false_tag: str = 'B'):
    count = 0
    is_correct_count = 0
    is_incorrect_count = 0
    is_not_attempted_count = 0
    is_judge_error_count = 0
    attempted_judge_count = 0
    details = []
    for i, j, k in zip(judged_answers, references, origial_responses):
        if i in [true_tag, false_tag]:
            attempted_judge_count += 1
        grade_letter = i
        detail = {
            'pred': k,
            'ref': j,
            'origin_grade_response': i,
            'grade_letter': grade_letter,
            'correct': False,
        }
        count += 1
        if grade_letter == true_tag:
            is_correct_count += 1
            detail['correct'] = True
        elif grade_letter == false_tag:
            is_incorrect_count += 1
        elif grade_letter == 'unknown':
            # The judge did not emit a usable grade. This is a *failure* of the
            # judge, surfaced separately so it cannot be silently read as a
            # wrong answer. It is still counted in not_attempted_count (as
            # upstream did) so every existing aggregate keeps its historical
            # meaning; judge_error_count is purely additive observability.
            is_judge_error_count += 1
            detail['judge_error'] = True
            is_not_attempted_count += 1
        else:
            is_not_attempted_count += 1
        details.append(detail)

    is_correct = is_correct_count / count
    is_incorrect = is_incorrect_count / count
    is_given_attempted = is_correct + is_incorrect
    accuracy_given_attempted = (is_correct / is_given_attempted
                                if is_given_attempted > 0 else 0)
    attempted_judge_ratio = attempted_judge_count / count

    f1 = (2 * accuracy_given_attempted * is_correct /
          (accuracy_given_attempted + is_correct) if
          (accuracy_given_attempted + is_correct) > 0 else 0)
    result = {
        metric_name: is_correct * 100,
        f'{metric_name}_given_attempted': accuracy_given_attempted * 100,
        'f1': f1,
        'attempted_ratio': attempted_judge_ratio * 100,
        'correct_count': is_correct_count,
        'incorrect_count': is_incorrect_count,
        'not_attempted_count': is_not_attempted_count,
        'judge_error_count': is_judge_error_count,
        'details': details,
    }
    return result


def _generic_llmjudge_postprocess(judgement: str,
                                  true_tag: str = 'A',
                                  false_tag: str = 'B'):
    # Single-direction conservative fix: only an explicitly anchored grade
    # ("grade: B", "grade is B", "final answer = A") is trusted. Everything
    # else keeps the legacy loose scan, so no judge reply that used to parse
    # now becomes an error (no regression, no silently-dropped samples).
    anchored = _ANCHORED_GRADE_RE.search(judgement)
    if anchored:
        return anchored.group(1).upper()
    match = re.search(rf'({re.escape(true_tag)}|{re.escape(false_tag)})',
                      judgement)
    grade_letter = match.group(0) if match else 'unknown'
    return grade_letter


@DICT_POSTPROCESSORS.register_module()
def generic_llmjudge_postprocess(
    output: dict,
    output_path: str,
    true_tag: str = 'A',
    false_tag: str = 'B',
) -> dict:

    judged_answers = []
    origial_responses = []
    references = []
    for k, v in output.items():
        origial_responses.append(v['prediction'])
        processed_judge = _generic_llmjudge_postprocess(
            v['prediction'], true_tag, false_tag)
        if processed_judge is not None:
            judged_answers.append(processed_judge)
            try:
                references.append(v['gold'])

            except KeyError:
                get_logger().warning(
                    f'No gold answer for {k}, use empty string as reference!')
                references.append('')
    results = get_final_results(judged_answers,
                                references,
                                origial_responses,
                                true_tag=true_tag,
                                false_tag=false_tag)
    results['details'] = output
    return results


def generic_llmjudge_academic_postprocess(
    output: dict,
    output_path: str,
    metric_name: str = 'accuracy',
) -> dict:
    judged_answers = []
    origial_responses = []
    references = []
    for k, v in output.items():
        origial_responses.append(v['prediction'])
        processed_judge = _generic_llmjudge_postprocess(v['prediction'])
        if processed_judge is not None:
            judged_answers.append(processed_judge)
            references.append(v['gold'])
    results = get_final_results(judged_answers, references, origial_responses,
                                metric_name)
    results['details'] = output
    # For academic summarizer
    results.pop('f1', None)
    return results
