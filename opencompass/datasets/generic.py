import re
from functools import lru_cache

from opencompass.registry import DICT_POSTPROCESSORS
from opencompass.utils import get_logger

# Grade cue immediately followed by a connector. The connector is mandatory so
# bare prose like "grade A of this study" (a quality qualifier) cannot open a
# verdict. Grammar covers word forms ("grade is B", "grade of A") and symbol
# forms ("grade: B", "grade = B", "grade:=B").
#
# Two words are deliberately absent from the cue list:
#
# - "final answer": in a judge reply it almost always names the *candidate's*
#   answer, not the grade. Treating it as a cue made
#   "The final answer was A, but it is wrong. Grade: B." parse as a correct
#   answer. Bare "answer" is excluded for the same family of reasons.
# - "be": ungrammatical in this slot, and it let "verdict beats any objection"
#   open a verdict on the "be" of "beats".
#
# The cue is matched case-insensitively; the grade letter is not.
_GRADE_CUE = (r'(?i:\b(?:grade|verdict)\s*'
              r'(?:(?:is|was|of)\s*[:=]?|[:=]{1,2}))')


@lru_cache(maxsize=8)
def _anchored_grade_re(true_tag: str, false_tag: str) -> re.Pattern:
    """Build the anchored grade pattern for one tag pair.

    Built from the caller's tags rather than hardcoding ``[AB]``: a caller
    configured with e.g. ``true_tag='A+'`` must keep working, and a hardcoded
    class would hand back a letter the aggregator then scores as *not
    attempted*.

    The surrounding prose is matched case-insensitively but the grade letter
    is matched case-sensitively and must not run into the next word. Without
    that, ``re.IGNORECASE`` lets the indefinite article in "Grade: ambiguous,
    ... B." be captured as the grade, and "Grade: Brilliant" is read as a "B"
    grade.

    Args:
        true_tag (`str`): The tag the caller treats as correct.
        false_tag (`str`): The tag the caller treats as incorrect.

    Returns:
        `re.Pattern`:
            A pattern whose group 1 is the anchored grade.
    """
    alts = '|'.join(
        re.escape(tag)
        for tag in sorted((true_tag, false_tag), key=len, reverse=True))
    return re.compile(_GRADE_CUE + r'\s*(?:[(\[]\s*)?(' + alts +
                      r')(?![A-Za-z0-9])\s*[)\]]?')


@lru_cache(maxsize=8)
def _bare_grade_re(true_tag: str, false_tag: str) -> re.Pattern:
    """Build the fallback pattern: a tag that is a *standalone* token.

    The historical scan took the first ``A``/``B`` character anywhere in the
    reply, which is what let prose donate a grade -- the ``A`` of ``GRADE: C``,
    the ``B`` of ``Verdict: Based on ...``, the ``B`` of ``Grade: Brilliant``.
    Requiring the tag to be its own token removes those while still finding a
    real trailing verdict such as the ``B`` in "leaning strongly B.".

    Args:
        true_tag (`str`): The tag the caller treats as correct.
        false_tag (`str`): The tag the caller treats as incorrect.

    Returns:
        `re.Pattern`:
            A pattern whose group 1 is the bare grade.
    """
    alts = '|'.join(
        re.escape(tag)
        for tag in sorted((true_tag, false_tag), key=len, reverse=True))
    return re.compile(r'(?<![A-Za-z0-9])(' + alts + r')(?![A-Za-z0-9])')


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
            # Uniform row shape: `judge_error` is present on every row rather
            # than only on failures. Note that
            # :func:`generic_llmjudge_postprocess` overwrites `details` with
            # the raw judge output afterwards, so this flag is visible to
            # direct callers of this function, not through that entry point.
            'judge_error': grade_letter not in (true_tag, false_tag),
        }
        count += 1
        if grade_letter == true_tag:
            is_correct_count += 1
            detail['correct'] = True
        elif grade_letter == false_tag:
            is_incorrect_count += 1
        else:
            # No usable grade: the judge failed to produce one. Counted as a
            # judge error *and* as not-attempted (as upstream did) so every
            # existing aggregate keeps its historical meaning.
            #
            # This changes no denominator. `accuracy` still divides by the
            # full sample count, so judge failures depress it. Read
            # `<metric>_given_attempted` for the figure that excludes them.
            # `judge_error_count` is the size of that excluded set; it
            # currently always equals `not_attempted_count`, and is named
            # separately so the two can diverge once a caller passes a grade
            # letter that is neither tag.
            is_judge_error_count += 1
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
    # Three rules, in order of trust:
    #
    # 1. An explicitly anchored grade wins, and the *last* one wins. A judge
    #    that revises itself ("draft grade was A ... final grade is B") means
    #    B, and the left-most match is not the verdict.
    # 2. The anchored letter is matched case-sensitively and must not run into
    #    the next word, so prose cannot donate a grade: "Grade: ambiguous, ...
    #    B." is a B, and "Grade: Brilliant" is not a B.
    # 3. Otherwise fall back to a bare scan -- but only of *standalone* tags.
    #    Requiring the letter to be its own token is what stops the scan
    #    inventing a grade out of the prose: the "A" of "GRADE: C", the "B" of
    #    "Verdict: Based on ...", the "B" of "Grade: Brilliant" are all parts
    #    of a word, while the "B" in "leaning strongly B." is a real grade and
    #    is still found.
    anchored = _anchored_grade_re(true_tag, false_tag).findall(judgement)
    if anchored:
        return anchored[-1]
    bare = _bare_grade_re(true_tag, false_tag).search(judgement)
    return bare.group(1) if bare else 'unknown'


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
