import csv
import os
import tempfile

import pytest

try:
    from opencompass.datasets.BuySideFinBench import BuySideFinBenchDataset
    HAS_OPENCOMPASS = True
except ImportError:
    HAS_OPENCOMPASS = False


def _make_csv(path, split, name, rows):
    """Helper to create a CSV file with standard BuySideFinBench format."""
    dirpath = os.path.join(path, split)
    os.makedirs(dirpath, exist_ok=True)
    filepath = os.path.join(dirpath, f'{name}.csv')
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'question', 'A', 'B', 'C', 'D', 'answer'])
        for i, row in enumerate(rows, 1):
            writer.writerow([i] + list(row))
    return filepath


@pytest.mark.skipif(not HAS_OPENCOMPASS,
                    reason='opencompass dependencies not installed')
class TestBuySideFinBenchDataset:
    """Tests for BuySideFinBenchDataset loader."""

    def _make_sample_rows(self, n):
        """Generate n sample question rows."""
        rows = []
        for i in range(n):
            rows.append((
                f'Question {i + 1}?',
                f'Option A{i + 1}',
                f'Option B{i + 1}',
                f'Option C{i + 1}',
                f'Option D{i + 1}',
                'A',
            ))
        return rows

    def test_load_basic(self):
        """Test that load() returns a DatasetDict with dev and test splits."""

        with tempfile.TemporaryDirectory() as tmpdir:
            dev_rows = self._make_sample_rows(5)
            test_rows = self._make_sample_rows(10)
            _make_csv(tmpdir, 'dev', 'test_subject', dev_rows)
            _make_csv(tmpdir, 'test', 'test_subject', test_rows)

            dataset = BuySideFinBenchDataset.load(tmpdir, 'test_subject')

            assert 'dev' in dataset
            assert 'test' in dataset
            assert len(dataset['dev']) == 5
            assert len(dataset['test']) == 10

    def test_field_names(self):
        """Test that loaded dataset contains the expected field names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = self._make_sample_rows(3)
            _make_csv(tmpdir, 'dev', 'test_subject', rows)
            _make_csv(tmpdir, 'test', 'test_subject', rows)

            dataset = BuySideFinBenchDataset.load(tmpdir, 'test_subject')

            expected_fields = {'question', 'A', 'B', 'C', 'D', 'answer'}
            assert set(dataset['dev'].column_names) == expected_fields
            assert set(dataset['test'].column_names) == expected_fields

    def test_field_values(self):
        """Test that field values are correctly loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [('What is 1+1?', '1', '2', '3', '4', 'B')]
            _make_csv(tmpdir, 'dev', 'test_subject', rows)
            _make_csv(tmpdir, 'test', 'test_subject', rows)

            dataset = BuySideFinBenchDataset.load(tmpdir, 'test_subject')

            sample = dataset['dev'][0]
            assert sample['question'] == 'What is 1+1?'
            assert sample['A'] == '1'
            assert sample['B'] == '2'
            assert sample['C'] == '3'
            assert sample['D'] == '4'
            assert sample['answer'] == 'B'

    def test_invalid_row_length(self):
        """Test that rows with wrong number of columns raise AssertionError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = os.path.join(tmpdir, 'dev')
            os.makedirs(dirpath, exist_ok=True)
            filepath = os.path.join(dirpath, 'test_subject.csv')
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['id', 'question', 'A', 'B', 'C', 'D', 'answer'])
                # Write a row with only 5 columns instead of 7
                writer.writerow([1, 'Q?', 'A', 'B', 'C'])

            with pytest.raises(AssertionError):
                BuySideFinBenchDataset.load(tmpdir, 'test_subject')

    def test_missing_file(self):
        """Test that missing CSV file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only dev, not test
            rows = self._make_sample_rows(5)
            _make_csv(tmpdir, 'dev', 'test_subject', rows)

            with pytest.raises(FileNotFoundError):
                BuySideFinBenchDataset.load(tmpdir, 'test_subject')

    def test_unicode_content(self):
        """Test that Chinese content is correctly loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [('某公司回购库存股1亿元，影响是？', '总资产减少', '总资产不变',
                     '总资产增加', '无影响', 'A')]
            _make_csv(tmpdir, 'dev', 'test_subject', rows)
            _make_csv(tmpdir, 'test', 'test_subject', rows)

            dataset = BuySideFinBenchDataset.load(tmpdir, 'test_subject')

            sample = dataset['dev'][0]
            assert '回购' in sample['question']
            assert sample['answer'] == 'A'


class TestCSVDataIntegrity:
    """Tests to verify the actual BuySideFinBench CSV data files."""

    DATA_DIR = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'BuySideFinBench')

    SUBJECTS = [
        'three_statements_zh', 'three_statements_en',
        'dcf_valuation_zh', 'dcf_valuation_en',
        'comps_analysis_zh', 'comps_analysis_en',
        'financial_ratios_zh', 'financial_ratios_en',
        'accounting_standards_zh', 'accounting_standards_en',
        'sensitivity_scenario_zh', 'sensitivity_scenario_en',
    ]

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(
                os.path.dirname(__file__), '..', '..', 'data',
                'BuySideFinBench', 'dev')),
        reason='BuySideFinBench data not found')
    @pytest.mark.parametrize('subject', SUBJECTS)
    def test_dev_csv_structure(self, subject):
        """Verify dev CSV has exactly 5 rows with valid answers."""
        filepath = os.path.join(self.DATA_DIR, 'dev', f'{subject}.csv')
        assert os.path.exists(filepath), f'Missing: {filepath}'
        with open(filepath, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ['id', 'question', 'A', 'B', 'C', 'D', 'answer']
            rows = list(reader)
            assert len(rows) == 5, f'{subject} dev should have 5 rows'
            for row in rows:
                assert len(row) == 7, f'Row has {len(row)} cols, expected 7'
                assert row[6] in ('A', 'B', 'C', 'D'), \
                    f'Invalid answer: {row[6]}'

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(
                os.path.dirname(__file__), '..', '..', 'data',
                'BuySideFinBench', 'test')),
        reason='BuySideFinBench data not found')
    @pytest.mark.parametrize('subject', SUBJECTS)
    def test_test_csv_structure(self, subject):
        """Verify test CSV has exactly 10 rows with valid answers."""
        filepath = os.path.join(self.DATA_DIR, 'test', f'{subject}.csv')
        assert os.path.exists(filepath), f'Missing: {filepath}'
        with open(filepath, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ['id', 'question', 'A', 'B', 'C', 'D', 'answer']
            rows = list(reader)
            assert len(rows) == 10, f'{subject} test should have 10 rows'
            for row in rows:
                assert len(row) == 7, f'Row has {len(row)} cols, expected 7'
                assert row[6] in ('A', 'B', 'C', 'D'), \
                    f'Invalid answer: {row[6]}'
