import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mmengine.config import ConfigDict

from opencompass.utils import result_station


def open_with_ascii_default(file, mode='r', *args, **kwargs):
    # Exercise a non-UTF-8 locale without changing the process-wide locale.
    if 'b' not in mode and 'encoding' not in kwargs:
        kwargs['encoding'] = 'ascii'
    return open(file, mode, *args, **kwargs)


class TestResultStation(unittest.TestCase):

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def prepare(self, subjective, text='中文回答🙂'):
        cfg = ConfigDict(work_dir=str(self.root / 'work'),
                         models=[dict(abbr='model')],
                         datasets=[dict(abbr='dataset')])
        model_result = 'model'
        if subjective:
            cfg.judge_models = [dict(abbr='judge')]
            cfg.datasets[0].base_models = [dict(abbr='base')]
            model_result = 'base_model_judged-by--judge'
        args = SimpleNamespace(station_path=str(self.root / 'station'),
                               station_overwrite=False,
                               mode='all')
        results = dict(accuracy=100.0, details=[dict(pred=text)])
        predictions = dict(prediction=text, gold=text)
        self.write(
            self.root / 'work' / 'results' / model_result / 'dataset.json',
            results)
        self.write(
            self.root / 'work' / 'predictions' / 'model' / 'dataset.json',
            {'0': predictions})
        archive = self.root / 'station' / 'dataset' / f'{model_result}.json'
        restored = (self.root / 'restored' / 'results' / model_result /
                    'dataset.json')
        return cfg, args, results, predictions, archive, restored

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False),
                        encoding='utf-8')

    def check_save(self, subjective, text='中文回答🙂'):
        cfg, args, results, predictions, archive, _ = self.prepare(
            subjective, text)
        with patch.object(result_station,
                          'open',
                          open_with_ascii_default,
                          create=True):
            self.assertTrue(result_station.save_to_station(cfg, args))
        saved = json.loads(archive.read_text(encoding='utf-8'))
        self.assertEqual(saved['results'], results)
        self.assertEqual(saved['predictions'], [predictions])

    def check_read(self, subjective, text='中文回答🙂'):
        cfg, args, results, predictions, archive, restored = self.prepare(
            subjective, text)
        self.write(archive, dict(results=results, predictions=[predictions]))
        cfg.work_dir = str(self.root / 'restored')
        with patch.object(result_station,
                          'open',
                          open_with_ascii_default,
                          create=True):
            entries = result_station.read_from_station(cfg, args)
        self.assertEqual(len(entries), 1)
        self.assertEqual(json.loads(restored.read_text(encoding='utf-8')),
                         results)

    def test_save_objective_unicode(self):
        self.check_save(False)

    def test_save_subjective_unicode(self):
        self.check_save(True)

    def test_read_objective_unicode(self):
        self.check_read(False)

    def test_read_subjective_unicode(self):
        self.check_read(True)

    def test_objective_ascii(self):
        self.check_save(False, 'answer')
        self.check_read(False, 'answer')

    def test_subjective_ascii(self):
        self.check_save(True, 'answer')
        self.check_read(True, 'answer')
