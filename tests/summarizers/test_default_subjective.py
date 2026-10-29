import json
from unittest.mock import MagicMock, patch

from mmengine.config import ConfigDict

from opencompass.summarizers.default_subjective import \
    DefaultSubjectiveSummarizer


def test_pick_up_results_keeps_non_numeric_diagnostics(tmp_path):
    origin = tmp_path / 'results' / 'mock_test' / 'wildbench.json'
    result_root = tmp_path / 'results'
    for base_model in ('base-a', 'base-b'):
        result_dir = result_root / (f'{base_model}_mock_test_judged-by--judge')
        result_dir.mkdir(parents=True)
        with open(result_dir / 'wildbench.json', 'w') as f:
            json.dump({'warning': 'no valid judgement'}, f)

    summarizer = DefaultSubjectiveSummarizer.__new__(
        DefaultSubjectiveSummarizer)
    summarizer.model_cfgs = [ConfigDict(abbr='mock_test')]
    summarizer.dataset_cfgs = [
        ConfigDict(
            abbr='wildbench',
            base_models=[ConfigDict(abbr='base-a'),
                         ConfigDict(abbr='base-b')],
            infer_cfg=ConfigDict(inferencer=ConfigDict(type='GenInferencer')),
        )
    ]
    summarizer.work_dir = str(tmp_path)
    summarizer.logger = MagicMock()

    with patch(
            'opencompass.summarizers.default_subjective.get_infer_output_path',
            return_value=str(origin)):
        raw, parsed, metrics, modes = summarizer._pick_up_results('judge')

    assert raw['mock_test']['wildbench']['warning'] == 'no valid judgement'
    assert parsed['mock_test'] == {}
    assert metrics == {}
    assert modes['wildbench'] == 'gen'
