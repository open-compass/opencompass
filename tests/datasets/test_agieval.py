import types
import unittest
from unittest.mock import patch

from opencompass.datasets.agieval.agieval import AGIEvalDataset_v2


class TestAGIEvalDataset(unittest.TestCase):

    def test_modelscope_labels_are_stringified(self):
        modelscope = types.ModuleType('modelscope')
        modelscope.MsDataset = types.SimpleNamespace(load=lambda *args, **kwargs: [
            {
                'passage': None,
                'question': 'numeric answer',
                'options': None,
                'label': '5',
                'answer': None,
            },
            {
                'passage': None,
                'question': 'latex answer',
                'options': None,
                'label': '$5$;$10$',
                'answer': None,
            },
            {
                'passage': None,
                'question': 'multiple choice answer',
                'options': ['A', 'B'],
                'label': "['A', 'B']",
                'answer': None,
            },
        ])

        with patch.dict('sys.modules', {'modelscope': modelscope}):
            with patch.dict('os.environ', {'DATASET_SOURCE': 'ModelScope'}):
                dataset = AGIEvalDataset_v2.load(
                    path='opencompass/agieval',
                    name='gaokao-mathcloze',
                    setting_name='zero-shot')

        self.assertEqual(dataset[0]['label'], '5')
        self.assertEqual(dataset[1]['label'], '$5$;$10$')
        self.assertEqual(dataset[2]['label'], 'AB')


if __name__ == '__main__':
    unittest.main()
