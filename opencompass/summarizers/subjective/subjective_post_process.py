import re


def post_process_autoj(judgement: str):
    """Input a string like below:

    xxx[[5]]xxx, and extract the score
    """
    pattern = r'\[(\d+)\]'
    matched_result = re.findall(pattern, judgement)
    if matched_result:
        score = int(matched_result[0])
    else:
        return None
    return {'score': score}


def post_process_judgelm(judgement: str):
    """Input a string like below:

    5, reason:xxx and extract the score
    """
    if len(judgement) >= 2:
        first_two_chars = judgement[:2]
        if first_two_chars.isdigit() and first_two_chars == '10':
            score = 10
        else:
            first_char = judgement[0]
            if first_char.isdigit() and 0 <= int(first_char) <= 9:
                score = int(first_char)
            else:
                return None
    elif len(judgement) == 1:
        if judgement.isdigit() and 0 <= int(judgement) <= 9:
            score = int(judgement)
        else:
            return None
    else:
        return None
    return {'score': score}
