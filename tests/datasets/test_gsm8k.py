from opencompass.datasets.gsm8k import gsm8k_postprocess


def test_gsm8k_postprocess_strips_thousands_separators():
    cases = {
        r'\boxed{$9{,}500}': '9500',
        r'\boxed{8,000}': '8000',
        'The answer is $9,500.': '9500',
        'The answer is 1,450,000': '1450000',
        'The total is $1,234,567.89': '1234567.89',
        r'Thus, Jill earns \(\boxed{\$57,500}\).': '57500',
        r'Josh incurred a loss: $$ \boxed{-10,000} $$': '-10000',
        r'The distance is \[ \boxed{180,000 \text{ meters}} \]': '180000',
        ('$22,500+$35,000 = $57,500 per year. '
         'The answer is $57,500'): '57500',
        'The answer is 2,000.\n\nQuestion: He has 5,000 coins.': '2000',
    }
    for text, expected in cases.items():
        assert gsm8k_postprocess(text) == expected, text


def test_gsm8k_postprocess_keeps_non_thousands_commas():
    # A comma is only a thousands separator when exactly three digits follow,
    # so lists, coordinates and decimal commas must not be merged.
    cases = {
        'The side lengths are 3,4,5': '5',
        'The point is (3,4)': '4',
        'Days 1,2 she ran 5 miles; total 12,15': '15',
        'So x = 2,5': '5',
        '12,34': '34',
        '1,2345': '2345',
        'Sides 3,4,5 give area 6, so it costs $1,200': '1200',
    }
    for text, expected in cases.items():
        assert gsm8k_postprocess(text) == expected, text
