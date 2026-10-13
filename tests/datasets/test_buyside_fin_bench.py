import pytest
from datasets import Dataset, DatasetDict

try:
    import opencompass.datasets.BuySideFinBench as buyside_module
    from opencompass.datasets.BuySideFinBench import (
        HF_REPO_ID, BuySideFinBenchDataset)
    HAS_OPENCOMPASS = True
except ImportError:
    HAS_OPENCOMPASS = False


@pytest.mark.skipif(not HAS_OPENCOMPASS,
                    reason='opencompass dependencies not installed')
class TestBuySideFinBenchDataset:
    """Tests for BuySideFinBenchDataset loader."""

    def _make_hf_dataset(self):
        return DatasetDict({
            'dev':
            Dataset.from_list([{
                'question': 'What is 1+1?',
                'A': '1',
                'B': '2',
                'C': '3',
                'D': '4',
                'answer': 'B',
            }]),
            'test':
            Dataset.from_list([{
                'question': '某公司回购库存股1亿元，影响是？',
                'A': '总资产减少',
                'B': '总资产不变',
                'C': '总资产增加',
                'D': '无影响',
                'answer': 'A',
            }]),
        })

    def test_load_uses_huggingface_dataset(self, monkeypatch):
        """Test that load() delegates to HuggingFace load_dataset."""
        calls = []

        def _fake_load_dataset(path, name):
            calls.append((path, name))
            return self._make_hf_dataset()

        monkeypatch.setattr(buyside_module, 'load_dataset',
                            _fake_load_dataset)

        dataset = BuySideFinBenchDataset.load(HF_REPO_ID, 'test_subject')

        assert calls == [(HF_REPO_ID, 'test_subject')]
        assert 'dev' in dataset
        assert 'test' in dataset
        assert len(dataset['dev']) == 1
        assert len(dataset['test']) == 1
        assert set(dataset['dev'].column_names) == {
            'question', 'A', 'B', 'C', 'D', 'answer'
        }

    def test_load_uses_default_repo_when_path_empty(self, monkeypatch):
        """Test that empty path falls back to the official HF repo id."""
        calls = []

        def _fake_load_dataset(path, name):
            calls.append((path, name))
            return self._make_hf_dataset()

        monkeypatch.setattr(buyside_module, 'load_dataset',
                            _fake_load_dataset)

        BuySideFinBenchDataset.load('', 'test_subject')

        assert calls == [(HF_REPO_ID, 'test_subject')]

    def test_field_values(self, monkeypatch):
        """Test that field values from HF datasets are preserved."""

        def _fake_load_dataset(path, name):
            return self._make_hf_dataset()

        monkeypatch.setattr(buyside_module, 'load_dataset',
                            _fake_load_dataset)

        dataset = BuySideFinBenchDataset.load(HF_REPO_ID, 'test_subject')

        dev_sample = dataset['dev'][0]
        assert dev_sample['question'] == 'What is 1+1?'
        assert dev_sample['B'] == '2'
        assert dev_sample['answer'] == 'B'
        test_sample = dataset['test'][0]
        assert '回购' in test_sample['question']
        assert test_sample['answer'] == 'A'

    def test_missing_split(self, monkeypatch):
        """Test that missing HF split raises a clear error."""

        def _fake_load_dataset(path, name):
            return DatasetDict({'dev': Dataset.from_list([])})

        monkeypatch.setattr(buyside_module, 'load_dataset',
                            _fake_load_dataset)

        with pytest.raises(ValueError, match='Missing split "test"'):
            BuySideFinBenchDataset.load(HF_REPO_ID, 'test_subject')
