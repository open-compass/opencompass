import unittest

from opencompass.datasets.GaokaoBench import GaokaoBenchEvaluator
from opencompass.datasets.kaoshi import KaoshiEvaluator


class TestMultiChoiceMarkerAtStart(unittest.TestCase):
    """The ``multi_choice`` postprocessors are meant to prefer the letters
    that follow the ``【答案】`` marker whenever the marker is present, and only
    fall back to scanning the tail of the output when it is absent.

    When the whitespace-stripped output *starts* with the marker,
    ``str.find`` returns ``0``; the ``answer_index > 0`` guard then wrongly
    treats the marker as absent and scans ``content[-10:]`` instead, dropping
    the real answer if it is not among the last 10 characters.
    """

    # Marker at index 0 after whitespace is stripped; the answer 'A' is
    # followed by explanatory text long enough to push it out of content[-10:].
    OUTPUT = '【答案】A，理由是经过仔细分析后得出的结论确实如此无误'

    def test_gaokao_bench_marker_at_start(self):
        ev = GaokaoBenchEvaluator(question_type='multi_choice')
        self.assertEqual(ev.do_predictions_postprocess(self.OUTPUT), ['A'])
        self.assertEqual(ev.score([self.OUTPUT], [['A']])['score'], 100.0)

    def test_kaoshi_marker_at_start(self):
        ev = KaoshiEvaluator(question_type='multi_choice')
        self.assertEqual(ev.do_predictions_postprocess(self.OUTPUT), ['A'])
        self.assertEqual(ev.score([self.OUTPUT], [['A']])['score'], 100.0)


if __name__ == '__main__':
    unittest.main()
