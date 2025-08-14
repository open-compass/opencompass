# flake8: noqa


def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except:
        return None


def last_n_boxed_strings(string, n):
    boxed_list = []

    work_str = string[:]
    while work_str and len(boxed_list) < n:
        idx = work_str.rfind('\\boxed')
        if idx < 0:
            idx = work_str.rfind('\\fbox')

        if idx < 0:
            break

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(work_str):
            if work_str[i] == '{':
                num_left_braces_open += 1
            elif work_str[i] == '}':
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is not None:
            boxed_expr = work_str[idx:right_brace_idx + 1]
            boxed_list.append(boxed_expr)
            work_str = work_str[:idx]
        else:
            work_str = work_str[:idx]

    boxed_list.reverse()
    return boxed_list


def get_answer_str(s: str, return_origin=False, num_answers=1):
    boxed_list = last_n_boxed_strings(s, num_answers)
    answer_list = [remove_boxed(b) if b else '' for b in boxed_list]

    missing = num_answers - len(answer_list)
    fill_str = s if return_origin else ''
    answer_list = [fill_str] * missing + answer_list

    return answer_list


def multi_box_extract_processor(solution: str,
                                math_mode='eval_peeking',
                                return_origin=False,
                                num_answers=1) -> tuple[bool, list | str]:
    answer = solution
    if math_mode == 'eval_peeking':
        answer = get_answer_str(solution, return_origin, num_answers)
    else:
        raise ValueError(f'Invalid math_mode: {math_mode}')
    return answer
