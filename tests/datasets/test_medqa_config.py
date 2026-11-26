"""Tests for the rule-based MedQA dataset config."""

from pathlib import Path

from mmengine.config import Config

from opencompass.utils.text_postprocessors import first_option_postprocess

CONFIG_PATH = (Path(__file__).resolve().parents[2] / 'opencompass' /
               'configs' / 'datasets' / 'MedQA' / 'MedQA_gen_3bf756.py')

# US and Mainland test questions have five options (A-E), Taiwan four.
EXPECTED_OPTIONS = {
    'MedQA_US': 'ABCDE',
    'MedQA_Mainland': 'ABCDE',
    'MedQA_Taiwan': 'ABCD',
}


def test_medqa_postprocessor_reads_every_option_letter():
    cfg = Config.fromfile(str(CONFIG_PATH))
    datasets = cfg['MedQA_datasets']
    assert [d['abbr'] for d in datasets] == list(EXPECTED_OPTIONS)
    for dataset in datasets:
        options = dataset['eval_cfg']['pred_postprocessor']['options']
        assert options == EXPECTED_OPTIONS[dataset['abbr']]
        for letter in options:
            reply = f'The findings point to one option.\nANSWER: {letter}'
            assert first_option_postprocess(reply, options) == letter
