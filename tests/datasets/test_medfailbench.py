import json
import tempfile
import unittest
from pathlib import Path

from opencompass.datasets.medfailbench import MedFailBenchDataset


class TestMedFailBenchDataset(unittest.TestCase):

    def test_loads_valid_jsonl(self):
        rows = [
            {
                'id': 'TRFAI001',
                'language': 'tr',
                'question': 'Test question?',
                'target': 'The answer should preserve urgent triage.',
                'clinical_domain': 'emergency',
                'risk_axis': 'rare_danger',
                'safety_gate': 'missed_urgent_escalation',
                'severity_1_to_5': 5,
                'metadata': {
                    'synthetic_only': True,
                    'contains_patient_data': False,
                },
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'medfailbench.jsonl'
            path.write_text(
                '\n'.join(json.dumps(row) for row in rows),
                encoding='utf-8')

            dataset = MedFailBenchDataset.load(str(path))

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]['id'], 'TRFAI001')
        self.assertEqual(dataset[0]['target'],
                         'The answer should preserve urgent triage.')

    def test_rejects_patient_data_flag(self):
        row = {
            'id': 'TRFAI001',
            'language': 'tr',
            'question': 'Test question?',
            'target': 'The answer should preserve urgent triage.',
            'clinical_domain': 'emergency',
            'risk_axis': 'rare_danger',
            'safety_gate': 'missed_urgent_escalation',
            'severity_1_to_5': 5,
            'metadata': {
                'synthetic_only': True,
                'contains_patient_data': True,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'medfailbench.jsonl'
            path.write_text(json.dumps(row), encoding='utf-8')

            with self.assertRaisesRegex(ValueError, 'patient-data-free'):
                MedFailBenchDataset.load(str(path))


if __name__ == '__main__':
    unittest.main()
