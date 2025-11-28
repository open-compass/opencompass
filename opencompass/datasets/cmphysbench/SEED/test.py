# test.py
# Run a battery of SEED examples across types.
# Usage:  python -m SEED.test

from __future__ import annotations

import warnings

warnings.filterwarnings('ignore',
                        category=SyntaxWarning)  # hide regex escape warnings

# Robust import whether executed as module or script
try:
    from .SEED import SEED
except Exception:
    import importlib
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.append(here)
    SEED = importlib.import_module('SEED').SEED  # function SEED in SEED.py


def run_case(idx: int, gt: str, pred: str, type: str, note: str = ''):
    print('\n' + '=' * 80)
    title = f'Case #{idx}  [{type}]'
    if note:
        title += f'  — {note}'
    print(title)
    print('-' * 80)
    try:
        score, rel_distance, tree_size, dist = SEED(gt, pred, type)
    except Exception as e:
        print(f'[ERROR] Exception while scoring: {e}')
        return

    print(f'GT LaTeX:      {gt}')
    print(f'Predicted:     {pred}')
    print(f'Score:         {score}')
    print(f'Rel Distance:  {rel_distance}')
    print(f'Tree Size:     {tree_size}')
    print(f'Raw Distance:  {dist}')


if __name__ == '__main__':
    # Test suite: a mix of Expression / Equation / Tuple / Interval / Numeric
    tests = [
        # ----------------------- Expression -----------------------
        {
            'type': 'Expression',
            'gt': r'2 m g + 4\frac{m v_0^2}{l}',
            'pred': r'2 m g + 4\frac{m v_0^2}{l}',
            'note': 'Exact match → expect 100'
        },
        {
            'type': 'Expression',
            'gt': r'2 m g + 4\frac{m v_0^2}{l}',
            'pred': r'2 m g + 2\frac{m v_0^2}{l}',
            'note': 'Coefficient differs → partial score'
        },

        # ----------------------- Equation -------------------------
        {
            'type': 'Equation',
            'gt': r'x^2 + 2x + 1 = 0',
            'pred': r'x^2 + 2x + 1 + 0 = 0',
            'note': 'Equivalent equation (add 0) → expect 100'
        },
        {
            'type': 'Equation',
            'gt': r'x + y = 0',
            'pred': r'x + y + 0 = 0',
            'note': 'Trivially equivalent → expect 100'
        },

        # ----------------------- Tuple ----------------------------
        {
            'type': 'Tuple',
            'gt': r'(x, y, z)',
            'pred': r'\left(x, y, z \right)',
            'note': 'Same tuple with \\left/\\right → expect 100'
        },
        {
            'type': 'Tuple',
            'gt': r'(x, y, z)',
            'pred': r'(x, z, y)',
            'note': 'Permutation in positions → partial score'
        },

        # ----------------------- Interval -------------------------
        {
            'type': 'Interval',
            'gt': r'[0, 1]',
            'pred': r'\left[0, 1 \right]',
            'note': 'Closed interval same form → expect 100'
        },
        {
            'type': 'Interval',
            'gt': r'(a, b]',
            'pred': r'[a, b]',
            'note': 'Open/closed boundary differs → likely < 100'
        },

        # ----------------------- Numeric (with units) -------------
        {
            'type': 'Numeric',
            'gt': r'4.2 \times 10^5 \mathrm{m^{2}}',
            'pred': r'0.42 \mathrm{km^{2}}',
            'note': 'Unit conversion m^2 ↔ km^2 → expect 100'
        },
        {
            'type': 'Numeric',
            'gt': r'1000 \mathrm{m}',
            'pred': r'1 \mathrm{km}',
            'note': 'Unit conversion m ↔ km → expect 100'
        },
        {
            'type': 'Numeric',
            'gt': r'9.81 \mathrm{m/s^{2}}',
            'pred': r'981 \mathrm{cm/s^{2}}',
            'note': 'Unit conversion m/s^2 ↔ cm/s^2 → expect 100'
        },
        {
            'type': 'Numeric',
            'gt': r'1.000 \mathrm{m}',
            'pred': r'0.990 \mathrm{m}',
            'note': '≈1% relative error → expect 80'
        },
        {
            'type': 'Numeric',
            'gt': r'3.14',
            'pred': r'3.1400',
            'note': 'No units, numerically equal → expect 100'
        },
        {
            'type': 'Numeric',
            'gt': r'5',
            'pred': r'-5',
            'note': 'Sign mismatch → expect 0'
        },
    ]

    for i, case in enumerate(tests, 1):
        run_case(i, case['gt'], case['pred'], case['type'],
                 case.get('note', ''))
