# flake8: noqa
from opencompass.datasets.MedCalc_Bench import check_correctness


def test_calid69_unparseable_ground_truth_does_not_crash():
    assert check_correctness('2 weeks, 3 days', 'no duration here', 69, None, None) == 0


def test_calid69_matching_duration():
    assert check_correctness('(2 weeks, 3 days)', '(2 weeks, 3 days)', 69, None, None) == 1
