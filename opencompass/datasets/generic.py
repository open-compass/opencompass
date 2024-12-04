import re


def get_final_results(judged_answers, references, origial_responses):
    count = 0
    is_correct_count = 0
    is_incorrect_count = 0
    is_not_attempted_count = 0
    details = []
    for i, j, k in zip(judged_answers, references, origial_responses):
        match = re.search(r'(A|B)', i)
        grade_letter = match.group(
            0) if match else 'B'  # Default to "INCORRECT" if no match
        detail = {
            'pred': k,
            'ref': j,
            'origin_grade_response': i,
            'grade_letter': grade_letter,
            'correct': False
        }
        count += 1
        if grade_letter == 'A':
            is_correct_count += 1
            detail['correct'] = True
        elif grade_letter == 'B':
            is_incorrect_count += 1
        else:
            is_not_attempted_count += 1
        details.append(detail)

    is_correct = is_correct_count / count
    is_incorrect = is_incorrect_count / count
    # is_not_attempted = is_not_attempted_count / count
    is_given_attempted = is_correct + is_incorrect
    accuracy_given_attempted = is_correct / is_given_attempted \
        if is_given_attempted > 0 else 0
    f1 = 2 * accuracy_given_attempted * is_correct / (
        accuracy_given_attempted + is_correct) if (accuracy_given_attempted +
                                                   is_correct) > 0 else 0
    result = {
        # 'accuracy_given_attempted': accuracy_given_attempted,
        'accuracy': accuracy_given_attempted * 100,
        'f1': f1,
        'details': details
    }
    return result


def _generic_llmjudge_postprocess(judgement: str):
    match = re.search(r'(A|B)', judgement)
    grade_letter = match.group(
        0) if match else 'B'  # Default to "INCORRECT" if no match
    return grade_letter


def generic_llmjudge_postprocess(
    output: dict,
    output_path: str,
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
    results = get_final_results(judged_answers, references, origial_responses)
    results['details'] = output
    return results
