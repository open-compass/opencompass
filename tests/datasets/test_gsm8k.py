from opencompass.datasets.gsm8k import gsm8k_postprocess


def test_gsm8k_postprocess_preserves_thousands_separators():
    cases = {
        r'\boxed{$9{,}500}': '9500',
        r'\boxed{8,000}': '8000',
        'The answer is $9,500.': '9500',
    }
    for text, expected in cases.items():
        assert gsm8k_postprocess(text) == expected
