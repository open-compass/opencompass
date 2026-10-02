"""Tests for the LLM judge postprocessor in opencompass/datasets/generic.py.

Three layers, matching how the parser actually decides:

1. an explicitly anchored grade ("grade: B", "grade is B", "verdict: A") is
   trusted, and the *last* anchor wins;
2. the grade letter is matched case-sensitively and must be a standalone
   token, so surrounding prose can never donate a grade;
3. with no anchor, a bare standalone tag is still accepted.

Most cases here are regressions: each one produced a confident WRONG grade,
or silently dropped a real one, under an earlier version of this parser. A
few (the bare-letter and "CORRECT" cases) are guards for legacy behaviour
that no version has ever broken.
"""

import unittest

from opencompass.datasets.generic import (_generic_llmjudge_postprocess,
                                          get_final_results)


class TestAnchoredGradeIsTrusted(unittest.TestCase):
    """An explicit, well-formed anchor is the grade."""

    def test_anchored_colon_grade(self):
        self.assertEqual(
            _generic_llmjudge_postprocess(
                'The answer is incorrect, grade: B. Summary: A clear miss.'),
            'B')

    def test_anchored_grade_is(self):
        self.assertEqual(
            _generic_llmjudge_postprocess(
                'The response is labeled letter A, but it should be marked '
                'incorrect, so the grade is B.'), 'B')

    def test_anchored_grade_of(self):
        self.assertEqual(
            _generic_llmjudge_postprocess('I give the response a grade of A.'),
            'A')

    def test_anchored_grade_equals(self):
        self.assertEqual(_generic_llmjudge_postprocess('grade = A'), 'A')

    def test_anchored_verdict(self):
        self.assertEqual(_generic_llmjudge_postprocess('My verdict is B.'),
                         'B')

    def test_bracketed_and_listed_forms(self):
        for text in ('grade: (B)', 'grade: [B]', '- Grade: B',
                     '1. Option A\n2. Option B\nGrade: B', 'Verdict\n\nis B'):
            self.assertEqual(_generic_llmjudge_postprocess(text), 'B', text)

    def test_grade_qualifier_is_not_a_grade(self):
        # "grade A of the study" names a quality level, not an assignment:
        # there is no connector after the cue, so the real anchor wins.
        self.assertEqual(
            _generic_llmjudge_postprocess(
                'This is grade A of the study quality, the verdict is B.'),
            'B')

    def test_bare_answer_is_an_option_not_a_grade(self):
        # Bare "answer" is not a cue: "the answer is A" names the option.
        self.assertEqual(
            _generic_llmjudge_postprocess(
                'The answer is A, and the model is correct, so the grade is '
                'B.'), 'B')


class TestLastAnchorWins(unittest.TestCase):
    """A judge that revises itself means its final statement."""

    def test_interim_grade_does_not_win(self):
        self.assertEqual(
            _generic_llmjudge_postprocess(
                'Grade: A. However, the justification contains a fatal '
                'error. Verdict: B.'), 'B')

    def test_self_correction_wins(self):
        self.assertEqual(
            _generic_llmjudge_postprocess(
                "The first draft's grade was A; after revision the final "
                'grade is B.'), 'B')

    def test_quoted_decoy_grade_does_not_win(self):
        # The judge quotes the student, then gives its own verdict.
        self.assertEqual(
            _generic_llmjudge_postprocess(
                'The student wrote "grade: A" in their draft. My own verdict '
                'is B.'), 'B')


class TestProseCannotDonateAGrade(unittest.TestCase):
    """The grade letter must be a real, standalone token.

    Each of these used to be captured out of the surrounding prose, turning a
    judge failure into a confident score.
    """

    def test_ignorecase_does_not_capture_the_article(self):
        # re.IGNORECASE used to make the "a" of "ambiguous" the grade.
        self.assertEqual(
            _generic_llmjudge_postprocess(
                'Grade: ambiguous, but leaning strongly B.'), 'B')
        self.assertEqual(
            _generic_llmjudge_postprocess(
                'Verdict: acceptable only after correction, ultimately B'),
            'B')

    def test_judge_failure_is_not_scored(self):
        for text in ('Verdict: a clear miss on the key claim.',
                     'Grade: bad, the conclusion contradicts the gold.',
                     'The final answer is: absolutely wrong on both counts.'):
            self.assertEqual(_generic_llmjudge_postprocess(text), 'unknown',
                             text)

    def test_letter_must_not_start_the_next_word(self):
        for text in ('Verdict: Based on the rubric, this is wrong.',
                     'Grade: able to answer two of three parts.',
                     'My verdict beats any objection.',
                     'The paper is a grade of art and craft.',
                     'Grade: Brilliant, yet ultimately wrong.', 'Grade: BEST'):
            self.assertEqual(_generic_llmjudge_postprocess(text), 'unknown',
                             text)

    def test_cue_word_itself_is_not_a_grade(self):
        # The historical scan read the "A" out of "GRADE" itself.
        for text in ('GRADE: C', 'GRADE: Inconclusive', 'FINAL ANSWER: C',
                     'Grade: C', 'Verdict: C', 'VERDICT: INCONCLUSIVE'):
            self.assertEqual(_generic_llmjudge_postprocess(text), 'unknown',
                             text)

    def test_final_answer_cue_reads_the_student(self):
        # "final answer" names the candidate's answer, so it must not be
        # treated as the grade -- these all used to return 'A'.
        for text in ('The final answer was A, but it is wrong. Grade: B.',
                     'The final answer is A, which is incorrect. Verdict: B.',
                     "The candidate's final answer is A. My grade: B."):
            self.assertEqual(_generic_llmjudge_postprocess(text), 'B', text)

    def test_final_answer_cue_cannot_win_a_later_verdict(self):
        # The cases above are also covered by last-anchor-wins, so they do
        # not by themselves prove the cue was removed. These put the real
        # verdict FIRST, so a "final answer" cue re-introduced into the list
        # would win and return 'A'.
        for text in ("My verdict is B. For reference, the candidate's final "
                     'answer was A.',
                     "Grade: B. The student's final answer is A."):
            self.assertEqual(_generic_llmjudge_postprocess(text), 'B', text)

    def test_letter_must_also_be_preceded_by_a_token_boundary(self):
        # The right-hand boundary alone is not enough: a tag glued to the end
        # of a word is not a standalone token either.
        for text in ('The checksum is SHA256A', 'model GPT4B output'):
            self.assertEqual(_generic_llmjudge_postprocess(text), 'unknown',
                             text)


class TestBareGradeStillAccepted(unittest.TestCase):
    """Replies with no anchor at all keep parsing."""

    def test_exact_letter_still_accepted(self):
        self.assertEqual(_generic_llmjudge_postprocess('A'), 'A')
        self.assertEqual(_generic_llmjudge_postprocess('  B  '), 'B')

    def test_bare_letter_in_prose(self):
        self.assertEqual(_generic_llmjudge_postprocess('Option A is correct'),
                         'A')
        self.assertEqual(_generic_llmjudge_postprocess('Answer: B'), 'B')

    def test_no_letter_is_unknown(self):
        self.assertEqual(_generic_llmjudge_postprocess('CORRECT'), 'unknown')


class TestConfigurableTags(unittest.TestCase):
    """The anchor is built from the caller's tags, not a hardcoded [AB]."""

    def test_multi_character_tags(self):
        self.assertEqual(
            _generic_llmjudge_postprocess('grade: A+', 'A+', 'B-'), 'A+')
        self.assertEqual(
            _generic_llmjudge_postprocess('grade: B-', 'A+', 'B-'), 'B-')

    def test_lowercase_tags(self):
        self.assertEqual(_generic_llmjudge_postprocess('grade: b', 'a', 'b'),
                         'b')

    def test_longer_tag_is_not_shadowed_by_a_prefix_tag(self):
        # The alternation is built longest-first, so a tag that is a prefix
        # of the other must not win just by being tried earlier.
        self.assertEqual(_generic_llmjudge_postprocess('Grade: A+', 'A', 'A+'),
                         'A+')
        self.assertEqual(_generic_llmjudge_postprocess('Grade: AB', 'A', 'AB'),
                         'AB')

    def test_word_tags(self):
        self.assertEqual(
            _generic_llmjudge_postprocess('grade: correct', 'correct',
                                          'incorrect'), 'correct')

    def test_anchored_tag_is_scored_not_dropped(self):
        # A correctly-anchored grade must reach the aggregator intact; a
        # hardcoded [AB] handed back 'A' and it was scored as not-attempted.
        res = get_final_results(['A+'], ['ref'], ['pred'],
                                true_tag='A+',
                                false_tag='B-')
        self.assertEqual(res['correct_count'], 1)
        self.assertEqual(res['not_attempted_count'], 0)
        self.assertEqual(res['judge_error_count'], 0)


class TestAggregation(unittest.TestCase):
    """get_final_results surfaces judge failures without shifting semantics."""

    def test_unknown_is_counted_and_keeps_existing_aggregates(self):
        judged = [
            'A', 'A', 'A', 'A', 'B', 'unknown', 'unknown', 'unknown', 'B', 'A'
        ]
        res = get_final_results(judged, ['r'] * 10, ['p'] * 10)
        # Accuracy semantics are unchanged from upstream: the denominator is
        # every sample, judge failures included.
        self.assertEqual(res['accuracy'], 50.0)
        # accuracy_given_attempted is the figure that excludes them.
        self.assertAlmostEqual(res['accuracy_given_attempted'], 100.0 * 5 / 7)
        self.assertEqual(res['not_attempted_count'], 3)
        self.assertEqual(res['judge_error_count'], 3)

    def test_every_detail_row_has_the_same_shape(self):
        # detail['judge_error'] must exist on graded rows too, or consumers
        # reading it get a KeyError on a ragged table.
        judged = ['A', 'B', 'unknown']
        res = get_final_results(judged, ['r'] * 3, ['p'] * 3)
        flags = [d['judge_error'] for d in res['details']]
        self.assertEqual(flags, [False, False, True])
        for d in res['details']:
            self.assertIn('judge_error', d)

    def test_unrecognised_grade_counts_as_judge_error(self):
        # A stray letter from a direct caller is a judge failure too.
        res = get_final_results(['A', 'C'], ['r'] * 2, ['p'] * 2)
        self.assertEqual(res['correct_count'], 1)
        self.assertEqual(res['not_attempted_count'], 1)
        self.assertEqual(res['judge_error_count'], 1)

    def test_no_judge_error_when_all_graded(self):
        judged = ['A', 'B', 'A', 'B', 'A']
        res = get_final_results(judged, ['r'] * 5, ['p'] * 5)
        self.assertEqual(res['judge_error_count'], 0)
        self.assertEqual(res['accuracy'], 60.0)


class TestJudgeTextToJudgeError(unittest.TestCase):
    """End-to-end: judge prose must actually reach the judge_error counter."""

    def test_anchored_non_binary_grade_is_a_judge_error(self):
        res = get_final_results(
            [_generic_llmjudge_postprocess('Grade: C')],
            ['ref'],
            ['Grade: C'],
        )
        self.assertEqual(res['judge_error_count'], 1)
        self.assertEqual(res['correct_count'], 0)
        self.assertEqual(res['incorrect_count'], 0)

    def test_all_caps_anchored_non_binary_grade_is_a_judge_error(self):
        res = get_final_results(
            [_generic_llmjudge_postprocess('GRADE: C')],
            ['ref'],
            ['GRADE: C'],
        )
        self.assertEqual(res['judge_error_count'], 1)
        self.assertEqual(res['correct_count'], 0)


if __name__ == '__main__':
    unittest.main()
