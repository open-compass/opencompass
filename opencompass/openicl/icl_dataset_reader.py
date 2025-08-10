"""Simple Dataset Reader."""

import random
from typing import Dict, List, Optional, Union

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.registry import ICL_DATASET_READERS
from opencompass.utils.types import (_check_dataset, _check_str,
                                     _check_type_list)


@ICL_DATASET_READERS.register_module()
class DatasetReader:
    """In-conext Learning Dataset Reader Class Generate an DatasetReader
    instance through 'dataset'.

    Attributes:
        dataset (:obj:`Dataset` or :obj:`DatasetDict`): The dataset to be read.
        input_columns (:obj:`List[str]` or :obj:`str`): A list of column names
            (a string of column name) in the dataset that represent(s) the
            input field.
        output_column (:obj:`str`): A column name in the dataset that
            represents the prediction field.
        input_template (:obj:`PromptTemplate`, optional): An instance of the
            :obj:`PromptTemplate` class, used to format the input field
            content during the retrieval process. (in some retrieval methods)
        output_template (:obj:`PromptTemplate`, optional): An instance of the
            :obj:`PromptTemplate` class, used to format the output field
            content during the retrieval process. (in some learnable retrieval
            methods)
        train_split (str): The name of the training split. Defaults to 'train'.
        train_range (int or float or str, optional): The size of the partial
            training dataset to load.
            If None, the entire training dataset will be loaded.
            If int or float, the random partial dataset will be loaded with the
            specified size.
            If str, the partial dataset will be loaded with the
            specified index list (e.g. "[:100]" for the first 100 examples,
            "[100:200]" for the second 100 examples, etc.). Defaults to None.
        test_split (str): The name of the test split. Defaults to 'test'.
        test_range (int or float or str, optional): The size of the partial
            test dataset to load.
            If None, the entire test dataset will be loaded.
            If int or float, the random partial dataset will be loaded with the
            specified size.
            If str, the partial dataset will be loaded with the
            specified index list (e.g. "[:100]" for the first 100 examples,
            "[100:200]" for the second 100 examples, etc.). Defaults to None.
    """
    dataset = None
    input_template = None
    output_template = None

    def __init__(self,
                 dataset: Union[Dataset, DatasetDict, str],
                 input_columns: Union[List[str], str],
                 output_column: Optional[str],
                 input_template: Optional[PromptTemplate] = None,
                 output_template: Optional[PromptTemplate] = None,
                 train_split: str = 'train',
                 train_range: Optional[Union[int, float, str]] = None,
                 test_split: str = 'test',
                 test_range: Optional[Union[int, float, str]] = None) -> None:
        self.input_columns = _check_type_list(input_columns, [List, str])
        if isinstance(self.input_columns, str):
            self.input_columns = self.input_columns.split()
        self.output_column = None
        if output_column:
            self.output_column = _check_str(output_column)

        train_range = _check_type_list(train_range, [None, int, float, str])
        test_range = _check_type_list(test_range, [None, int, float, str])

        if input_template is not None:
            self.input_template = PromptTemplate._check_prompt_template(
                input_template)
        if output_template is not None:
            self.output_template = PromptTemplate._check_prompt_template(
                output_template)

        self.dataset = _check_dataset(dataset)
        if isinstance(self.dataset, Dataset):
            self.dataset = DatasetDict({
                'train': self.dataset,
                'test': self.dataset
            })

        # Normalize the dataset so that it has only "train" and "test" splits.
        for origin_split, mapped_split, split_range in [[
                train_split, 'train', train_range
        ], [test_split, 'test', test_range]]:
            self.dataset[mapped_split] = load_partial_dataset(
                self.dataset[origin_split], size=split_range)

    def generate_input_field_prompt(self, entry: Dict) -> str:
        """Generate a prompt for the input field based on the provided
        :obj:`entry` data.

        Args:
            entry (:obj:`Dict`): A piece of data to be used for generating the
                prompt.

        Returns:
            :obj:`str`: The generated prompt.
        """
        prompt = None
        if self.input_template is None:
            prompt = ' '.join([str(entry[ctx]) for ctx in self.input_columns])
        else:
            prompt = self.input_template.generate_item(entry)
        return prompt

    def generate_input_field_corpus(self,
                                    dataset: Union[Dataset, DatasetDict],
                                    split: Optional[str] = None) -> List[str]:
        """Generate corpus for input field.

        Args:
            dataset (:obj:`Dataset` or :obj:`DatasetDict`): A
                :obj:`datasets.Dataset` or :obj:`datasets.DatasetDict`
                instance.
            split (:obj:`str`, optional): The split of the dataset to use. If
                :obj:`None`, the entire dataset will be used. Defaults to
                ``None``.

        Returns:
            :obj:`List[str]`: A list of generated input field prompts.
        """
        if split is not None:
            dataset = dataset[split]
        corpus = []
        for entry in dataset:
            corpus.append(self.generate_input_field_prompt(entry))
        return corpus

    def generate_output_field_prompt(self, entry: Dict) -> str:
        """Generate a prompt for the output field based on the provided
        :obj:`entry` data.

        Args:
            entry (:obj:`Dict`): A piece of data to be used for generating the
            prompt.

        Returns:
            :obj:`str`: The generated prompt.
        """
        prompt = None
        if self.output_template is None:
            prompt = str(entry[self.output_column])
        else:
            prompt = self.output_template.generate_item(entry)
        return prompt

    def generate_output_field_corpus(self,
                                     dataset: Union[Dataset, DatasetDict],
                                     split: Optional[str] = None) -> List[str]:
        """Generate corpus for output field.

        Args:
            dataset (:obj:`Dataset` or :obj:`DatasetDict`): A
                :obj:`datasets.Dataset` or :obj:`datasets.DatasetDict`
                instance.
            split (:obj:`str`, optional): The split of the dataset to use.
                If :obj:`None`, the entire dataset will be used. Defaults to
                ``None``.

        Returns:
            :obj:`List[str]`: A list of generated output field prompts.
        """
        if split is not None:
            dataset = dataset[split]
        corpus = []
        for entry in dataset:
            corpus.append(self.generate_output_field_prompt(entry))
        return corpus

    def generate_input_output_field_prompt(self, entry: Dict) -> str:
        """Generate a prompt for the input-output field based on the
        provided:obj:`entry` data.

        Args:
            entry (:obj:`Dict`): A piece of data to be used for generating the
            prompt.

        Returns:
            :obj:`str`: The generated prompt.
        """
        prompt = None
        if self.input_output_template is None:
            prompt = ' '.join([entry[ctx] for ctx in self.input_columns] +
                              [str(entry[self.output_column])])
        else:
            prompt = self.input_output_template.generate_item(entry)
        return prompt

    def _check_dataset_reader(obj) -> 'DatasetReader':
        if isinstance(obj, DatasetReader):
            return obj
        else:
            raise TypeError(f'Expected a DatasetReader object, but got {obj}')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __repr__(self):
        return (f'DatasetReader({{\n    dataset: {self.dataset},'
                f'\n    input_columns: {self.input_columns},\n'
                f'    output_columns: {self.output_column}\n}})')


def load_partial_dataset(
        dataset: Dataset,
        size: Optional[Union[int, float, str]] = None) -> Dataset:
    """Load a partial dataset.

    Args:
        dataset (Dataset): A :obj:`datasets.Dataset` instance.
        size (int or float or (int, int), optional): The size of the partial
            dataset to load. If None, the entire dataset will be loaded.
            If int or float, the random partial dataset will be loaded with the
            specified size. If str, the partial dataset will be loaded with the
            specified index list (e.g. "[:100]" for the first 100 examples,
            "[100:200]" for the second 100 examples, etc.). Defaults to None.
    """
    total_size = len(dataset)
    index_list = list(range(total_size))
    if isinstance(size, (int, float)):
        if size >= total_size or size <= 0:
            return dataset
        if size > 0 and size < 1:
            size = int(size * total_size)
        rand = random.Random(x=size)
        rand.shuffle(index_list)
        dataset = dataset.select(index_list[:size])
    elif isinstance(size, str):
        dataset = dataset.select(eval(f'index_list{size}'))
    return dataset


class DatasetEncoder(torch.utils.data.Dataset):

    def __init__(self,
                 datalist: List,
                 model_name=None,
                 tokenizer=None) -> None:
        self.datalist = datalist
        if model_name is None and tokenizer is None:
            raise ValueError('model_name and tokenizer could not both be None')
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = 'left'
        self.encode_dataset = []
        self.init_dataset()
        self.datalist_length = len(self.encode_dataset)

    def init_dataset(self):
        for idx, data in enumerate(self.datalist):
            tokenized_data = self.tokenizer.encode_plus(data,
                                                        truncation=True,
                                                        return_tensors='pt',
                                                        verbose=False)
            self.encode_dataset.append({
                'input_ids':
                tokenized_data.input_ids[0],
                'attention_mask':
                tokenized_data.attention_mask[0],
                'metadata': {
                    'id': idx,
                    'len': len(tokenized_data.input_ids[0]),
                    'text': data
                }
            })

    def __len__(self):
        return self.datalist_length

    def __getitem__(self, idx):
        return self.encode_dataset[idx]
