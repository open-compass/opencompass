"""Tests for the LLM judge postprocessor in opencompass/datasets/generic.py.

These cover the anchored-grade extraction (a judge's free-text reply may name
non-grade things that look like letters) and the aggregation contract (judge
failures are surfaced, never silently absorbed into accuracy).

Design intent: the *anchored* grade ("grade is B", "grade: B", "final answer
is: B", "grade of A") is the only thing trusted when it is present. Everything
else falls back to the historical loose scan so no previously-parseable judge
reply regresses to 'unknown'. Bare qualifiers ("grade A of this study") and
bare option labels ("the answer is A") must never be mistaken for a grade.
"""

import unittest

from opencompass.datasets.generic import (
    _generic_llmjudge_postprocess,
    get_final_results,
)


class TestGenericLlmJudgeGradeExtraction(unittest.TestCase):
    """Anchored grades are extracted even when prose letters precede them."""

    def test_prose_first_letter_no_longer_wins(self):
        # Upstream regex took the first A/B in the reply; here the prose
        # letter 'A' must lose to the anchored "grade is B".
        self.assertEqual(
            _generic_llmjudge_postprocess(
                'As a judge, I considered the evidence carefully. '
                'The final grade is B.'),
            'B')

    def test_anchored_colon_grade(self):
        self.assertEqual(
            _generic_llmjudge_postprocess(
                'The answer is incorrect, grade: B. Summary: A clear miss.'),
            'B')

    def test_anchored_grade_is(self):
        self.assertEqual(
            _generic_llmjudge_postprocess(
                'The response is labeled letter A, but it should be marked '
                'incorrect, so the grade is B.'),
            'B')

    def test_anchored_final_answer_colon(self):
        self.assertEqual(
            _generic_llmjudge_postprocess('The final answer is: B'), 'B')

    def test_anchored_grade_of(self):
        self.assertEqual(
            _generic_llmjudge_postprocess('I give the response a grade of A.'),
            'A')

    def test_anchored_grade_equals(self):
        self.assertEqual(_generic_llmjudge_postprocess('grade = A'), 'A')

    def test_anchored_verdict(self):
        self.assertEqual(
            _generic_llmjudge_postprocess('My verdict is B.'), 'B')

    def test_exact_letter_still_accepted(self):
        self.assertEqual(_generic_llmjudge_postprocess('A'), 'A')
        self.assertEqual(_generic_llmjudge_postprocess('  B  '), 'B')

    def test_no_letter_is_unknown(self):
        self.assertEqual(_generic_llmjudge_postprocess('CORRECT'), 'unknown')


class TestGenericLlmJudgeNoFalseAnchor(unittest.TestCase):
    """Qualifiers and option labels must not be mistaken for a grade."""

    def test_grade_a_of_is_qualifier_not_grade(self):
        # "grade A of the study" is a quality qualifier, not an assignment.
        # The real verdict is the anchored "verdict is B".
        self.assertEqual(
            _generic_llmjudge_postprocess(
                'This is grade A of the study quality, the verdict is B.'),
            'B')

    def test_bare_answer_is_a_option_not_grade(self):
        # "the answer is A" names the option; the grade is anchored later.
        self.assertEqual(
            _generic_llmjudge_postprocess(
                'The answer is A, and the model is correct, so the grade is B.'),
            'B')


class TestGenericLlmJudgeAggregation(unittest.TestCase):
    """get_final_results surfaces judge failures without shifting semantics."""

    def test_unknown_is_surfaced_and_keeps_not_attempted_semantics(self):
        judged = ['A', 'A', 'A', 'A', 'B',
                  'unknown', 'unknown', 'unknown', 'B', 'A']
        res = get_final_results(judged, ['r'] * 10, ['p'] * 10)
        # 5 correct / 10 total — accuracy semantics unchanged from upstream.
        self.assertEqual(res['accuracy'], 50.0)
        # upstream counted 'unknown' as not_attempted; keep that, and ADD the
        # judge-failure surface so failures cannot be read as wrong answers.
        self.assertEqual(res['not_attempted_count'], 3)
        self.assertEqual(res['judge_error_count'], 3)
        for detail in res['details']:
            is_unknown = detail['grade_letter'] == 'unknown'
            self.assertEqual('judge_error' in detail, is_unknown)

    def test_no_judge_error_when_all_graded(self):
        judged = ['A', 'B', 'A', 'B', 'A']
        res = get_final_results(judged, ['r'] * 5, ['p'] * 5)
        self.assertEqual(res['judge_error_count'], 0)
        self.assertEqual(res['accuracy'], 60.0)


if __name__ == '__main__':
    unittest.main()
