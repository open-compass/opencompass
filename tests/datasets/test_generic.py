from opencompass.datasets.generic import generic_llmjudge_postprocess


def test_generic_llmjudge_postprocess_custom_tags():
    output = {
        '0': {
            'prediction': 'X',
            'gold': 'B',
        },
        '1': {
            'prediction': 'Y',
            'gold': 'C',
        },
    }

    scores = generic_llmjudge_postprocess(
        output,
        output_path='unused.json',
        true_tag='X',
        false_tag='Y',
    )

    assert scores['accuracy'] == 50
    assert scores['accuracy_given_attempted'] == 50
    assert scores['attempted_ratio'] == 100
    assert scores['correct_count'] == 1
    assert scores['incorrect_count'] == 1
    assert scores['not_attempted_count'] == 0
