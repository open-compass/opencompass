"""Unit tests for SemanticSimilarityEvaluator (G4, PR-4).

Covers the five test categories the extension guide requires: normal
input, empty input, parse failure, boundary values, and multiple
references. The sentence-transformers encoder is replaced by a
deterministic fake so no model is downloaded.
"""

import unittest

import numpy as np

from opencompass.openicl.icl_evaluator import ICL_EVALUATORS
from opencompass.openicl.icl_evaluator.icl_semantic_similarity_evaluator \
    import SemanticSimilarityEvaluator

# A tiny deterministic "embedding space": every distinct word maps to an
# orthogonal unit vector, so cosine similarity is 1.0 iff the word sets
# are identical, and shrinks as they diverge.
_WORDS = ['蓝色', '天空', '晴朗', '大海', '森林', '绿色']


def _fake_encode(texts):
    vectors = []
    for text in texts:
        vec = np.zeros(len(_WORDS))
        for word in str(text).split():
            if word in _WORDS:
                vec[_WORDS.index(word)] = 1.0
        norm = np.linalg.norm(vec)
        vectors.append(vec / norm if norm else vec)
    return np.array(vectors)


def _make_evaluator(**kwargs):
    evaluator = SemanticSimilarityEvaluator(**kwargs)
    evaluator._encode = _fake_encode
    return evaluator


class TestRegistration(unittest.TestCase):

    def test_registered(self):
        self.assertIs(
            ICL_EVALUATORS.get('SemanticSimilarityEvaluator'),
            SemanticSimilarityEvaluator)


class TestNormalInput(unittest.TestCase):

    def test_similarity_in_range_and_mean(self):
        evaluator = _make_evaluator()
        result = evaluator.score(['蓝色 天空', '绿色 森林'],
                                 ['蓝色 天空', '绿色 森林'])

        self.assertAlmostEqual(result['semantic_similarity'], 100.0)
        self.assertEqual(len(result['details']), 2)
        self.assertAlmostEqual(result['details'][0]['similarity'], 1.0)

    def test_partial_overlap_scores_between_zero_and_hundred(self):
        evaluator = _make_evaluator()
        result = evaluator.score(['蓝色 天空'], ['蓝色 大海'])

        similarity = result['semantic_similarity']
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 100.0)


class TestEmptyInput(unittest.TestCase):

    def test_empty_lists_return_error(self):
        evaluator = _make_evaluator()

        result = evaluator.score([], [])

        self.assertIn('error', result)

    def test_length_mismatch_returns_error(self):
        evaluator = _make_evaluator()

        result = evaluator.score(['蓝色 天空'],
                                 ['蓝色 天空', '绿色 森林'])

        self.assertIn('error', result)


class TestParseFailure(unittest.TestCase):

    def test_none_and_non_string_inputs_do_not_crash(self):
        evaluator = _make_evaluator(postprocess=False)

        result = evaluator.score([None, 123],
                                 ['蓝色 天空', '绿色 森林'])

        self.assertNotIn('error', result)
        self.assertEqual(len(result['details']), 2)


class TestBoundaryValues(unittest.TestCase):

    def test_identical_texts_reach_one(self):
        evaluator = _make_evaluator()
        result = evaluator.score(['晴朗 蓝色 天空'],
                                 ['晴朗 蓝色 天空'])

        self.assertAlmostEqual(result['semantic_similarity'], 100.0)

    def test_disjoint_texts_score_zero(self):
        evaluator = _make_evaluator()
        result = evaluator.score(['晴朗'], ['大海'])

        self.assertAlmostEqual(result['semantic_similarity'], 0.0)


class TestMultipleReferences(unittest.TestCase):

    def test_best_reference_wins(self):
        evaluator = _make_evaluator()

        # The prediction matches the second reference exactly, so the
        # max over references must reach 1.0 even though the first does
        # not.
        result = evaluator.score(['绿色 森林'],
                                 [['蓝色 天空', '绿色 森林']])

        self.assertAlmostEqual(result['details'][0]['similarity'], 1.0)

    def test_threshold_reports_semantic_accuracy(self):
        evaluator = _make_evaluator(threshold=0.9)
        result = evaluator.score(
            ['蓝色 天空', '绿色 森林'],  # exact match / mismatch
            ['蓝色 天空', '蓝色 天空'])

        self.assertAlmostEqual(result['semantic_accuracy'], 50.0)


if __name__ == '__main__':
    unittest.main()
