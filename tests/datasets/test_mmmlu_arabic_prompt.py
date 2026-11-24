"""Unit tests for Arabic MMMLU and MMMLU-Lite prompts."""

import pathlib
import unittest

from opencompass.configs.datasets.mmmlu.mmmlu_prompt import (
    get_few_shot_prompts_ar, )


class TestMMMLUArabicPrompts(unittest.TestCase):
    """Ensure Arabic MMMLU prompts use consistent A-D labels and valid shots."""

    def test_few_shot_arabic_prompt_has_option_d_and_no_cot_mismatch(self):
        shots = get_few_shot_prompts_ar('HINT', 'PROMPT')
        human_turns = [m['prompt'] for m in shots if m['role'] == 'HUMAN']
        # All 5 few-shot examples must ask for A, B, C, or D without truncation
        for idx, turn in enumerate(human_turns[:5]):
            self.assertIn('D.', turn, f'Shot {idx} missing option D')
            self.assertNotIn('أو  المدرسة', turn)
            self.assertNotIn('لنفكر خطوة بخطوة', turn)
            self.assertNotIn('نفكر خطوة بخطوة', turn)

    def test_zero_shot_arabic_configs_use_latin_option_labels(self):
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        for rel in [
                'opencompass/configs/datasets/mmmlu/mmmlu_gen_c51a84.py',
                'opencompass/configs/datasets/mmmlu_lite/mmmlu_lite_gen_c51a84.py',
        ]:
            content = (repo_root / rel).read_text(encoding='utf-8')
            self.assertNotIn(r'\nأ. {{A}}', content)
            self.assertIn(r'A. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}', content)


if __name__ == '__main__':
    unittest.main()
