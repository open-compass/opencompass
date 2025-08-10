"""Basic Retriever."""
from abc import abstractmethod
from typing import Dict, List, Optional

from mmengine.dist import is_main_process

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.utils.prompt import PromptList


class BaseRetriever:
    """Base class for In-context Learning Example Retriever, without any
    retrieval method implemented.

    Args:
        dataset (`BaseDataset`): Any BaseDataset instances.
            Attributes of ``reader``, ``train`` and ``test`` will be used.
        ice_separator (`Optional[str]`): The separator between each in-context
            example template when origin `PromptTemplate` is provided. Defaults
            to '\n'.
        ice_eos_token (`Optional[str]`): The end of sentence token for
            in-context example template when origin `PromptTemplate` is
            provided. Defaults to '\n'.
        ice_num (`Optional[int]`): The number of in-context example template
            when origin `PromptTemplate` is provided. Defaults to 1.
    """
    index_ds = None
    test_ds = None

    def __init__(self,
                 dataset,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 ice_num: Optional[int] = 1) -> None:
        self.ice_separator = ice_separator
        self.ice_eos_token = ice_eos_token
        self.ice_num = ice_num
        self.is_main_process = is_main_process()
        self.dataset_reader = dataset.reader
        self.index_ds = dataset.train
        self.test_ds = dataset.test

    @abstractmethod
    def retrieve(self) -> List[List[int]]:
        """Retrieve the in-context example index for each test example."""

    def get_labels(
            self,
            ice_template: Optional[PromptTemplate] = None,
            prompt_template: Optional[PromptTemplate] = None) -> List[str]:
        """Get the labels of the dataset, especially useful for ppl inferencer.
        If `ice_template` is provided, the labels will be the keys of the
        template. If `prompt_template` is provided, the labels will be the keys
        of the template. If neither of them is provided, the labels will be the
        unique values of the output column.

        Args:
            ice_template (`Optional[PromptTemplate]`): The template for
                in-context example. Defaults to None.
            prompt_template (`Optional[PromptTemplate]`): The template for
                prompt. Defaults to None.
        """
        if prompt_template is not None and isinstance(prompt_template.template,
                                                      Dict):
            labels = list(prompt_template.template.keys())
        elif ice_template is not None and ice_template.ice_token is not None \
                and isinstance(ice_template.template, Dict):
            labels = list(ice_template.template.keys())
        else:
            labels = list(set(self.test_ds[self.dataset_reader.output_column]))
        return labels

    def generate_ice(self,
                     idx_list: List[int],
                     ice_template: Optional[PromptTemplate] = None) -> str:
        """Generate the in-context example for one test example. If
        `ice_template` is an instance of `PromptTemplate`, the `ice_separator`
        and `ice_eos_token` will be set as empty.

        Args:
            idx_list (`List[int]`): The index of in-context examples for the
                test example.
            ice_template (`Optional[PromptTemplate]`): The template for
                in-context example. Defaults to None.
        """
        if ice_template is None:
            assert len(
                idx_list
            ) == 0, 'You have not specified ice_template while retrieving examples from train set! Please either specify ice_template or use `ZeroRetriever`.'  # noqa

        if ice_template is not None and ice_template.prompt_type == 'meta':
            ice_separator, ice_eos_token = '', ''
        else:
            ice_separator = self.ice_separator
            ice_eos_token = self.ice_eos_token

        generated_ice_list = []
        for idx in idx_list:
            generated_ice_list.append(
                ice_template.generate_ice_item(
                    self.index_ds[idx],
                    self.index_ds[idx][self.dataset_reader.output_column]))
        if len(generated_ice_list) > 0 and isinstance(generated_ice_list[0],
                                                      PromptList):
            generated_ice = []
            for ice in generated_ice_list:
                generated_ice += ice + ice_separator
            generated_ice.append(ice_eos_token)
        else:
            generated_ice = ice_separator.join(
                generated_ice_list) + ice_eos_token
        return generated_ice

    def generate_label_prompt(self,
                              idx: int,
                              ice: str,
                              label,
                              ice_template: Optional[PromptTemplate] = None,
                              prompt_template: Optional[PromptTemplate] = None,
                              remain_sep: Optional[bool] = False) -> str:
        """Generate the prompt for one test example in perpelxity evaluation
        with `prompt_template`. If `prompt_template` is not provided, the
        `ice_template` will be used to generate the prompt.

        Args:
            idx (`int`): The index of the test example.
            ice (`str`): The in-context example for the test example.
            label (`str`): The label of the test example.
            ice_template (`Optional[PromptTemplate]`): The template for
                in-context example. Defaults to None.
            prompt_template (`Optional[PromptTemplate]`): The template for
                prompt. Defaults to None.
            remain_sep (`Optional[bool]`): Whether to remain the sep token.
                Defaults to False.
        """
        if prompt_template is not None and ice_template is not None:
            if prompt_template.ice_token is not None:
                return prompt_template.generate_label_prompt_item(
                    self.test_ds[idx], ice, label, remain_sep)
            else:
                raise NotImplementedError(
                    'ice_token of prompt_template is not provided')
        elif ice_template is not None and prompt_template is None:
            if ice_template.ice_token is not None:
                return ice_template.generate_label_prompt_item(
                    self.test_ds[idx], ice, label, remain_sep)
            else:
                raise NotImplementedError(
                    'ice_token of ice_template is not provided')
        elif ice_template is None and prompt_template is not None:
            return prompt_template.generate_label_prompt_item(
                self.test_ds[idx], ice, label, remain_sep)
        else:
            raise NotImplementedError(
                'Leaving prompt as empty is not supported')

    def generate_prompt_for_generate_task(
            self,
            idx,
            ice,
            gen_field_replace_token='',
            ice_template: Optional[PromptTemplate] = None,
            prompt_template: Optional[PromptTemplate] = None):
        """Generate the prompt for one test example in generative evaluation
        with `prompt_template`. If `prompt_template` is not provided, the
        `ice_template` will be used to generate the prompt. The token
        represented by `gen_field_replace_token` will not be replaced by the
        generated text, or it will leaks the answer.

        Args:
            idx (`int`): The index of the test example.
            ice (`str`): The in-context example for the test example.
            gen_field_replace_token (`str`): The token of the answer in the
                prompt. Defaults to ''.
            ice_template (`Optional[PromptTemplate]`): The template for
                in-context example. Defaults to None.
            prompt_template (`Optional[PromptTemplate]`): The template for
                prompt. Defaults to None.
        """
        if prompt_template is not None and ice_template is not None:
            if prompt_template.ice_token is not None:
                return prompt_template.generate_item(
                    self.test_ds[idx],
                    output_field=self.dataset_reader.output_column,
                    output_field_replace_token=gen_field_replace_token,
                    ice_field_replace_token=ice)
            else:
                raise NotImplementedError(
                    'ice_token of prompt_template is not provided')
        elif ice_template is not None and prompt_template is None:
            if ice_template.ice_token is not None:
                return ice_template.generate_item(
                    self.test_ds[idx],
                    output_field=self.dataset_reader.output_column,
                    output_field_replace_token=gen_field_replace_token,
                    ice_field_replace_token=ice)
            else:
                raise NotImplementedError(
                    'ice_token of ice_template is not provided')
        elif ice_template is None and prompt_template is not None:
            return prompt_template.generate_item(
                self.test_ds[idx],
                output_field=self.dataset_reader.output_column,
                output_field_replace_token=gen_field_replace_token,
                ice_field_replace_token=ice)
        else:
            raise NotImplementedError(
                'Leaving prompt as empty is not supported')

    def generate_prompt_and_label_for_generate_task(
            self,
            idx,
            ice,
            gen_field_replace_token='',
            ice_template: Optional[PromptTemplate] = None,
            prompt_template: Optional[PromptTemplate] = None):
        """Generate the prompt and the label info for one test example in
        generative evaluation with `prompt_template`. If `prompt_template` is
        not provided, the `ice_template` will be used to generate the prompt.
        The token represented by `gen_field_replace_token` will not be replaced
        by the generated text, or it will leaks the answer.

        Args:
            idx (`int`): The index of the test example.
            ice (`str`): The in-context example for the test example.
            gen_field_replace_token (`str`): The token of the answer in the
                prompt. Defaults to ''.
            ice_template (`Optional[PromptTemplate]`): The template for
                in-context example. Defaults to None.
            prompt_template (`Optional[PromptTemplate]`): The template for
                prompt. Defaults to None.
        """
        if prompt_template is not None and ice_template is not None:
            if prompt_template.ice_token is not None:
                return prompt_template.generate_item(
                    self.test_ds[idx],
                    output_field=self.dataset_reader.output_column,
                    output_field_replace_token=gen_field_replace_token,
                    ice_field_replace_token=ice), self.test_ds[idx]['label']
            else:
                raise NotImplementedError(
                    'ice_token of prompt_template is not provided')
        elif ice_template is not None and prompt_template is None:
            if ice_template.ice_token is not None:
                return ice_template.generate_item(
                    self.test_ds[idx],
                    output_field=self.dataset_reader.output_column,
                    output_field_replace_token=gen_field_replace_token,
                    ice_field_replace_token=ice), self.test_ds[idx]['label']
            else:
                raise NotImplementedError(
                    'ice_token of ice_template is not provided')
        elif ice_template is None and prompt_template is not None:
            return prompt_template.generate_item(
                self.test_ds[idx],
                output_field=self.dataset_reader.output_column,
                output_field_replace_token=gen_field_replace_token,
                ice_field_replace_token=ice), self.test_ds[idx]['label']
        else:
            raise NotImplementedError(
                'Leaving prompt as empty is not supported')

    def generate_prompt_for_adv_generate_task(
            self,
            idx,
            ice,
            extra_prompt=dict(),
            gen_field_replace_token='',
            ice_template: Optional[PromptTemplate] = None,
            prompt_template: Optional[PromptTemplate] = None):
        """Generate the prompt for one test example in generative evaluation
        with `prompt_template`. If `prompt_template` is not provided, the
        `ice_template` will be used to generate the prompt. The token
        represented by `gen_field_replace_token` will not be replaced by the
        generated text, or it will leaks the answer.

        Args:
            idx (`int`): The index of the test example.
            ice (`str`): The in-context example for the test example.
            gen_field_replace_token (`str`): The token of the answer in the
                prompt. Defaults to ''.
            ice_template (`Optional[PromptTemplate]`): The template for
                in-context example. Defaults to None.
            prompt_template (`Optional[PromptTemplate]`): The template for
                prompt. Defaults to None.
        """
        if prompt_template is not None and ice_template is not None:
            if prompt_template.ice_token is not None:
                return prompt_template.generate_item(
                    {
                        **self.test_ds[idx],
                        **extra_prompt
                    },
                    output_field=self.dataset_reader.output_column,
                    output_field_replace_token=gen_field_replace_token,
                    ice_field_replace_token=ice)
            else:
                raise NotImplementedError(
                    'ice_token of prompt_template is not provided')
        elif ice_template is not None and prompt_template is None:
            if ice_template.ice_token is not None:
                return ice_template.generate_item(
                    {
                        **self.test_ds[idx],
                        **extra_prompt
                    },
                    output_field=self.dataset_reader.output_column,
                    output_field_replace_token=gen_field_replace_token,
                    ice_field_replace_token=ice)
            else:
                raise NotImplementedError(
                    'ice_token of ice_template is not provided')
        elif ice_template is None and prompt_template is not None:
            return prompt_template.generate_item(
                {
                    **self.test_ds[idx],
                    **extra_prompt
                },
                output_field=self.dataset_reader.output_column,
                output_field_replace_token=gen_field_replace_token,
                ice_field_replace_token=ice)
        else:
            raise NotImplementedError(
                'Leaving prompt as empty is not supported')
