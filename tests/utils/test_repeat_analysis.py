"""Unit tests for opencompass.utils.repeat_analysis."""

import types
import unittest

import tiktoken

from opencompass.utils import repeat_analysis as ra


class _FakeBosHFTokenizer:
    """Minimal HuggingFace-style tokenizer stub that prepends a BOS token."""

    is_fast = True

    def __init__(self, split_think: bool = False):
        self.split_think = split_think

    def _encode_one(self, text: str, add_special_tokens: bool) -> list:
        tokens = []
        for word in text.split():
            if self.split_think and word == '</think>':
                tokens.extend(['</', 'think', '>'])
            else:
                tokens.append(word)
        ids = [100 + (abs(hash(t)) % 10000) for t in tokens]
        self._id_to_tok = getattr(self, '_id_to_tok', {1: '<bos>'})
        for i, t in zip(ids, tokens):
            self._id_to_tok[i] = t
        return ([1] + ids) if add_special_tokens else ids

    def __call__(self, text, add_special_tokens: bool = True):
        if isinstance(text, list):
            return {
                'input_ids': [
                    self._encode_one(t, add_special_tokens) for t in text
                ]
            }
        return types.SimpleNamespace(
            input_ids=self._encode_one(text, add_special_tokens))

    def convert_ids_to_tokens(self, ids):
        return [self._id_to_tok.get(i, str(i)) for i in ids]


class TestRepeatAnalysis(unittest.TestCase):
    """Regression tests for repeat_analysis bug fixes."""

    def test_non_repeating_sequence_returns_empty_pattern(self):
        clean_tokens = [f'tok_{i}' for i in range(120)]
        stats = ra._repeat_pattern_stats(clean_tokens)
        self.assertEqual(stats['repeat_pattern_count'], 0)
        self.assertEqual(stats['repeat_pattern_ratio'], 0)
        self.assertEqual(stats['repeat_pattern'], '')

    def test_think_tag_with_bos_hf_tokenizer(self):
        wrapper = ra.TokenizerWrapper(_FakeBosHFTokenizer(split_think=False),
                                      'hf', 'fake-bos')
        records = [{
            'model': 'm1',
            'benchmark': 'b1',
            'sample_id': '0',
            'prediction_path': '/tmp/pred.json',
            'text': 'step1 step2 step3 </think> final answer',
            'res_length': None,
        }]
        ra._analyze_benchmark(records, {'m1': wrapper}, think_tag='</think>')
        # Tokens: ['<bos>', 'step1', 'step2', 'step3', '</think>', 'final', 'answer']
        self.assertEqual(records[0]['reasoning_length'], 4)
        self.assertEqual(records[0]['content_length'], 2)

    def test_think_tag_multitoken_tiktoken_ignores_other_closing_tags(self):
        tok = tiktoken.encoding_for_model('gpt-4o')
        wrapper = ra.TokenizerWrapper(tok, 'tiktoken', 'gpt-4o')
        think_tokens = wrapper.tokenize('</think>', add_special_tokens=False)
        self.assertGreater(len(think_tokens), 1)
        # Case A: BPE merges preceding space into '</' (' </', 'think', '>')
        prefix = '<div>hello</div> reasoning steps '
        suffix = ' final answer'
        text_spaced = prefix + '</think>' + suffix
        records = [{
            'model': 'm1',
            'benchmark': 'b1',
            'sample_id': '0',
            'prediction_path': '/tmp/pred.json',
            'text': text_spaced,
            'res_length': None,
        }]
        ra._analyze_benchmark(records, {'m1': wrapper}, think_tag='</think>')
        self.assertEqual(
            records[0]['reasoning_length'],
            len(wrapper.tokenize(prefix, add_special_tokens=True)),
        )
        self.assertEqual(
            records[0]['content_length'],
            len(wrapper.tokenize(suffix, add_special_tokens=False)),
        )

        # Case B: Text with '</div>' only (no '</think>') must not split
        no_think = '<div>hello</div> plain answer without think tag'
        rec_no_think = [{
            'model': 'm1',
            'benchmark': 'b1',
            'sample_id': '1',
            'prediction_path': '/tmp/pred.json',
            'text': no_think,
            'res_length': None,
        }]
        ra._analyze_benchmark(rec_no_think, {'m1': wrapper},
                              think_tag='</think>')
        self.assertEqual(rec_no_think[0]['reasoning_length'], 0)
        self.assertEqual(rec_no_think[0]['content_length'],
                         len(wrapper.tokenize(no_think)))

    def test_threshold_boundary_count_equal_min_is_flagged(self):
        record = {
            'model': 'm1',
            'benchmark': 'b1',
            'sample_id': '0',
            'prediction_path': '/tmp/pred.json',
            'length': 800,
            'repeat_analysis_skipped': False,
            'repeat_pattern_ratio': 0.8,
            'repeat_pattern': 'a b c',
            'repeat_pattern_count': ra.MIN_PATTERN_COUNT,
            'gzip_compression_ratio': 12.0,
        }
        stats = {
            'b1': {
                'repeat_pattern_ratio': {
                    'threshold': ra.MIN_PATTERN_COUNT
                },
                'gzip_compression_ratio': {
                    'threshold': ra.GZIP_THRESHOLD
                },
                'length': {
                    'p75': 400
                },
            }
        }
        abnormal = ra._build_abnormal_samples([record], stats)
        self.assertEqual(len(abnormal['repeat_pattern']), 1)

    def test_format_summary_table_handles_empty_benchmark_stats(self):
        report = {'benchmark_stats': {}, 'abnormal_samples': {}}
        table = ra._format_analysis_summary_table(report)
        self.assertGreaterEqual(len(table), 2)
        self.assertEqual(table[-1][0], 'Overall')
        self.assertEqual(table[-1][1], '0')


if __name__ == '__main__':
    unittest.main()
