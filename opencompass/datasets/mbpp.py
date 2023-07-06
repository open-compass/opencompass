import contextlib
import io
import re
import signal

from datasets import DatasetDict, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MBPPDataset(BaseDataset):

    @staticmethod
    def load(path: str):

        def processing_test(example):
            example['test_case'] = example['test_list']
            example['test_list'] = '\n'.join(example['test_list'])
            example['test_list_2'] = example['test_list']
            return example

        train = load_dataset('json', data_files=path,
                             split='train[:10]').map(processing_test)
        test = load_dataset('json', data_files=path,
                            split='train[10:510]').map(processing_test)
        return DatasetDict({'train': train, 'test': test})


class TimeOutException(Exception):
    pass


@ICL_EVALUATORS.register_module()
class MBPPEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        assert len(predictions) == len(references)
        predictions = [self._process_answer(pred) for pred in predictions]

        result = {'pass': 0, 'timeout': 0, 'failed': 0, 'wrong_answer': 0}
        for test_case, pred in zip(references, predictions):
            programs = self._process_test(test_case, pred)
            try:
                with self.swallow_io():
                    with self.time_limit(2):
                        exec(programs)
                result['pass'] += 1
            except TimeOutException:
                result['timeout'] += 1
            except AssertionError:
                result['wrong_answer'] += 1
            except BaseException:
                result['failed'] += 1

        result['score'] = result['pass'] / len(predictions) * 100
        return result

    def _process_answer(self, text):
        text = text.strip()
        match = re.search(r"('\s*|)(\[DONE\]|DONE)", text)
        if match:
            text = text[:match.start()]
        match = re.search(r"(\[BEGIN\]|BEGIN)('\s*|)", text)
        if match:
            text = text[match.end():]
        text = text.strip()
        if text.startswith("'"):
            text = text[1:]
        if text.endswith("'"):
            text = text[:-1]
        return text

    def _process_test(self, test_case, pred):
        formatted = pred + '\n'
        formatted += test_case
        return formatted

    @contextlib.contextmanager
    def swallow_io(self):
        stream = self.WriteOnlyStringIO()
        with contextlib.redirect_stdout(stream):
            with contextlib.redirect_stderr(stream):
                with self.redirect_stdin(stream):
                    yield

    @contextlib.contextmanager
    def time_limit(self, seconds: float):

        def signal_handler(signum, frame):
            raise TimeOutException('Time out!')

        signal.setitimer(signal.ITIMER_REAL, seconds)
        signal.signal(signal.SIGALRM, signal_handler)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    class WriteOnlyStringIO(io.StringIO):
        """StringIO that throws an exception when it's read from."""

        def read(self, *args, **kwargs):
            raise IOError

        def readline(self, *args, **kwargs):
            raise IOError

        def readlines(self, *args, **kwargs):
            raise IOError

        def readable(self, *args, **kwargs):
            """Returns True if the IO object can be read."""
            return False

    class redirect_stdin(contextlib._RedirectStream):  # type: ignore
        _stream = 'stdin'
