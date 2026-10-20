import copy
import itertools
from typing import Callable, List, Optional, Union

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator

from .arc import ARCDataset
from .ceval import CEvalDataset
from .cmmlu import CMMLUDataset
from .commonsenseqa import commonsenseqaDataset
from .hellaswag import HellaswagDataset_V2
from .mmlu import MMLUDataset
from .obqa import OBQADataset
from .piqa import PIQADatasetV2
from .race import RaceDataset
from .siqa import SiqaDatasetV3
from .xiezhi import XiezhiDataset


def get_origin_patterns(option_keys):
    return [tuple(option_keys)]


def get_circular_patterns(option_keys):
    double_option_keys = option_keys + option_keys
    circular_patterns = [
        tuple(double_option_keys[i:i + len(option_keys)])
        for i in range(len(option_keys))
    ]
    return circular_patterns


def get_all_possible_patterns(option_keys):
    circular_patterns = list(itertools.permutations(option_keys))
    return circular_patterns


class CircularDatasetMeta(type):
    """This Meta Class is designed to transform a class that reads datasets
    into one that supports reading datasets required for CircularEval. It
    overloads an existing load method for the original class.

    The Meta Class should possess the following attributes:

    - `dataset_class` (class): The class for reading datasets, such as
        `CEvalDataset`.
    - `default_circular_splits` (list, optional): The default splits of the
        dataset that need to undergo CircularEval, like ['val', 'test']. If a
        `Dataset` is loaded originally, this field will be ignored.
    - `default_option_keys` (list): The keys for options in the dataset, such
        as ['A', 'B', 'C', 'D'].
    - `default_answer_key` (str, optional): The key for answers in the dataset,
        like 'answer'. This is an alternative to
        `default_answer_key_switch_method`.
    - `default_answer_key_switch_method` (function, optional): The method to
        transform the key for answers in the dataset. This is an alternative to
        `default_answer_key`.
    """

    @staticmethod
    def make_circular_items(
        origin_item,
        circular_patterns,
        option_keys,
        answer_key,
        answer_key_switch_method,
        qid,
    ):
        items = []
        for circular_pattern in circular_patterns:
            item = copy.deepcopy(origin_item)
            for i in range(len(option_keys)):
                item[circular_pattern[i]] = origin_item[option_keys[i]]
            if answer_key_switch_method is None:
                if origin_item[answer_key] in option_keys:
                    item[answer_key] = circular_pattern[option_keys.index(
                        origin_item[answer_key])]
                else:
                    pass
            else:
                item = answer_key_switch_method(item, circular_pattern)
            item['qid'] = qid
            item['circular_pattern'] = tuple(circular_pattern)
            items.append(item)
        return items

    @staticmethod
    def make_circular_dataset(dataset, circular_patterns, option_keys,
                              answer_key, answer_key_switch_method):
        circulated_items = []
        for i, item in enumerate(dataset):
            item = CircularDatasetMeta.make_circular_items(
                item,
                circular_patterns,
                option_keys,
                answer_key,
                answer_key_switch_method,
                i,
            )
            circulated_items.extend(item)
        return Dataset.from_list(circulated_items)

    def make_circular(
        dataset: Union[Dataset, DatasetDict],
        circular_splits: Optional[List[str]] = ['test'],
        circular_patterns: str = 'circular',
        option_keys: List[str] = ['A', 'B', 'C', 'D'],
        answer_key: Optional[str] = 'answer',
        answer_key_switch_method: Optional[Callable] = None,
    ):
        """Transform the dataset into one that is compatible with CircularEval.
        In CircularEval, the original multiple-choice questions with options
        ABCD are augmented by shuffling the order of options, such as BCDA,
        CDAB, DABC, etc. A model is considered correct only if it answers all
        augmented questions correctly. This method effectively prevents models
        from memorizing answers.

        Args:
        datasets: The dataset to be augmented.
        circular_splits: List of splits to make circular. This is only
            effective when the dataset is a DatasetDict.
        circular_patterns: Method for circular processing, can be 'circular'
            for single cycle or 'all_possible' for all permutations, default
            is 'circular'.
        option_keys: List of keys for options, default to ['A', 'B', 'C', 'D'].
        answer_key: Key for the answer, default to 'answer'. When specified,
            ensure that the content of answer_key is among the option_keys.
            It is an alternative to specifying answer_key_switch_method.
        answer_key_switch_method: Function to modify the answer_key. It is an
            alternative to specifying answer_key.
        """

        if isinstance(circular_patterns, str):
            if circular_patterns == 'circular':
                circular_patterns = get_circular_patterns(option_keys)
            elif circular_patterns == 'all_possible':
                circular_patterns = get_all_possible_patterns(option_keys)
            else:
                raise ValueError(
                    f'Unknown circular_patterns: {circular_patterns}')
        else:
            assert isinstance(circular_patterns, list)
            assert all([isinstance(i, list) for i in circular_patterns])
            # TODO: other necessary sanity checks
            raise NotImplementedError(
                'circular_patterns int list of list has not been tested yet')

        if answer_key is None and answer_key_switch_method is None:
            raise ValueError(
                'answer_key and answer_key_switch_method cannot be both None')
        if answer_key is not None and answer_key_switch_method is not None:
            raise ValueError(
                'either answer_key or answer_key_switch_method should be None')

        if isinstance(dataset, Dataset):
            dataset = CircularDatasetMeta.make_circular_dataset(
                dataset,
                circular_patterns,
                option_keys,
                answer_key,
                answer_key_switch_method,
            )
        else:
            assert isinstance(dataset, DatasetDict)
            dataset_dict = {}
            for split in dataset:
                if circular_splits is not None and split in circular_splits:
                    dataset_dict[
                        split] = CircularDatasetMeta.make_circular_dataset(
                            dataset[split],
                            circular_patterns,
                            option_keys,
                            answer_key,
                            answer_key_switch_method,
                        )
                else:
                    dataset_dict[split] = dataset[split]
            dataset = DatasetDict(dataset_dict)
        return dataset

    def __new__(cls, name, bases, dct):
        new_cls = super().__new__(cls, name, bases, dct)

        def load(cls, circular_patterns='circular', *args, **kwargs):
            circular_splits = getattr(cls, 'default_circular_splits', None)
            option_keys = getattr(cls, 'default_option_keys', None)
            if 'option_keys' in kwargs:
                option_keys = kwargs.pop('option_keys')
            assert option_keys is not None, 'option_keys cannot be None'
            answer_key = getattr(cls, 'default_answer_key', None)
            if 'answer_key' in kwargs:
                answer_key = kwargs.pop('answer_key')
            answer_key_switch_method = getattr(
                cls, 'default_answer_key_switch_method', None)
            dataset = cls.dataset_class.load(*args, **kwargs)
            return CircularDatasetMeta.make_circular(
                dataset,
                circular_splits,
                circular_patterns,
                option_keys,
                answer_key,
                answer_key_switch_method,
            )

        setattr(new_cls, 'load', classmethod(load))
        return new_cls


class CircularCEvalDataset(CEvalDataset, metaclass=CircularDatasetMeta):
    dataset_class = CEvalDataset
    default_circular_splits = ['val', 'test']
    default_option_keys = ['A', 'B', 'C', 'D']
    default_answer_key = 'answer'


class CircularMMLUDataset(MMLUDataset, metaclass=CircularDatasetMeta):
    dataset_class = MMLUDataset
    default_circular_splits = ['test']
    default_option_keys = ['A', 'B', 'C', 'D']
    default_answer_key = 'target'


class CircularCMMLUDataset(CMMLUDataset, metaclass=CircularDatasetMeta):
    dataset_class = CMMLUDataset
    default_circular_splits = ['test']
    default_option_keys = ['A', 'B', 'C', 'D']
    default_answer_key = 'answer'


class CircularCSQADataset(commonsenseqaDataset, metaclass=CircularDatasetMeta):
    dataset_class = commonsenseqaDataset
    default_circular_splits = ['validation']
    default_option_keys = ['A', 'B', 'C', 'D', 'E']
    default_answer_key = 'answerKey'


class CircularARCDataset(ARCDataset, metaclass=CircularDatasetMeta):
    dataset_class = ARCDataset
    default_circular_splits = None
    default_option_keys = ['textA', 'textB', 'textC', 'textD']

    def default_answer_key_switch_method(item, circular_pattern):
        circular_pattern = tuple(i[-1] for i in circular_pattern)
        item['answerKey'] = circular_pattern['ABCD'.index(item['answerKey'])]
        return item


class CircularHSWAGDataset(HellaswagDataset_V2, metaclass=CircularDatasetMeta):
    dataset_class = HellaswagDataset_V2
    default_circular_splits = None
    default_option_keys = ['A', 'B', 'C', 'D']
    default_answer_key = 'label'


class CircularOBQADataset(OBQADataset, metaclass=CircularDatasetMeta):
    dataset_class = OBQADataset
    default_circular_splits = None
    default_option_keys = ['A', 'B', 'C', 'D']
    default_answer_key = 'answerKey'


class CircularRaceDataset(RaceDataset, metaclass=CircularDatasetMeta):
    dataset_class = RaceDataset
    default_circular_splits = ['test']
    default_option_keys = ['A', 'B', 'C', 'D']
    default_answer_key = 'answer'


class CircularXiezhiDataset(XiezhiDataset, metaclass=CircularDatasetMeta):
    dataset_class = XiezhiDataset
    default_circular_splits = None
    default_option_keys = ['A', 'B', 'C', 'D']
    default_answer_key = 'answer'


class CircularsiqaDataset(SiqaDatasetV3, metaclass=CircularDatasetMeta):
    dataset_class = SiqaDatasetV3
    default_circular_splits = ['validation']
    default_option_keys = ['A', 'B', 'C']
    default_answer_key = 'answer'


class CircularPIQADataset(PIQADatasetV2, metaclass=CircularDatasetMeta):
    dataset_class = PIQADatasetV2
    default_circular_splits = ['validation']
    default_option_keys = ['sol1', 'sol2']

    def default_answer_key_switch_method(item, circular_pattern):
        circular_pattern = tuple(int(i[-1]) - 1 for i in circular_pattern)
        item['answer'] = 'AB'[circular_pattern['AB'.index(item['answer'])]]
        return item


class CircularEvaluator(BaseEvaluator):
    """This Evaluator assesses datasets post-Circular processing, generating
    the following evaluation metrics:

    - `acc_{origin|circular|all_possible}`: Treats each question with shuffled
        answer options as separate, calculating accuracy.
    - `perf_{origin|circular|all_possible}`: According Circular logic, a
        question is considered correct only if all its variations with shuffled
        options are answered correctly, calculating accuracy. perf is short for
        perfect.
    - `more_{num}_{origin|circular|all_possible}`: According to Circular logic,
        a question is considered correct only if the number of its variations
        answered correctly is greater than or equal to `num`, calculating
        accuracy.

    Note that when the `all_possible` method is used to shuffle option order,
        it naturally includes the Circular method, and its metrics will also be
        output.

    Args:
        circular_pattern: The method of shuffling options, either 'circular' or
            'all_possible', defaulting to 'circular'.
    """

    def __init__(self, circular_pattern='circular'):
        super().__init__()
        self.circular_pattern = circular_pattern

    def score(self, predictions, references, test_set):
        circular_patterns = {}
        circular_patterns['origin'] = get_origin_patterns(
            test_set[0]['circular_pattern'])
        circular_patterns['circular'] = get_circular_patterns(
            test_set[0]['circular_pattern'])
        if self.circular_pattern == 'all_possible':
            circular_patterns['all_possible'] = get_all_possible_patterns(
                test_set[0]['circular_pattern'])

        metrics = {}
        tmp_metrics = {}
        tmp_metrics.update({f'correct_{k}': 0 for k in circular_patterns})
        tmp_metrics.update({f'count_{k}': 0 for k in circular_patterns})
        # calculate the original accuracy
        for pred, refr, origin_item in zip(predictions, references, test_set):
            circular_pattern = origin_item['circular_pattern']
            for k in circular_patterns:
                if tuple(circular_pattern) in circular_patterns[k]:
                    tmp_metrics[f'correct_{k}'] += 1 if pred == refr else 0
                    tmp_metrics[f'count_{k}'] += 1

        for k in circular_patterns:
            metrics[f'acc_{k}'] = (tmp_metrics[f'correct_{k}'] /
                                   tmp_metrics[f'count_{k}'] * 100)

        # calculate the circular accuracy
        _details = {k: {} for k in circular_patterns}
        for pred, refr, origin_item in zip(predictions, references, test_set):
            index = origin_item['qid']
            circular_pattern = origin_item['circular_pattern']
            for k in circular_patterns:
                if tuple(circular_pattern) in circular_patterns[k]:
                    _details[k].setdefault(
                        index, []).append(True if pred == refr else False)
        for k in _details:
            _details[k] = {
                index: sum(_details[k][index])
                for index in _details[k]
            }
        for k in _details:
            for j in range(1, len(circular_patterns[k]) + 1):
                count = sum([_details[k][index] >= j for index in _details[k]])
                total = len(_details[k])
                if j != len(circular_patterns[k]):
                    metrics[f'more_{j}_{k}'] = count / total * 100
                else:
                    metrics[f'perf_{k}'] = count / total * 100

        return metrics
