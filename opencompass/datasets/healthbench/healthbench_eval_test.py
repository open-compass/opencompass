from .healthbench_eval import RubricItem, calculate_score


def test_calculate_score():
    rubric_items = [
        RubricItem(criterion='test', points=7, tags=[]),
        RubricItem(criterion='test', points=5, tags=[]),
        RubricItem(criterion='test', points=10, tags=[]),
        RubricItem(criterion='test', points=-6, tags=[]),
    ]
    grading_response_list = [
        {
            'criteria_met': True
        },
        {
            'criteria_met': False
        },
        {
            'criteria_met': True
        },
        {
            'criteria_met': True
        },
    ]
    total_possible = 7 + 5 + 10
    achieved = 7 + 0 + 10 - 6
    assert (calculate_score(rubric_items, grading_response_list) == achieved /
            total_possible)


if __name__ == '__main__':
    test_calculate_score()
