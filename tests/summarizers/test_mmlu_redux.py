"""Tests for the MMLU-Redux summarizer configuration."""

import os
from collections import Counter

from mmengine.config import Config

from opencompass.configs.summarizers.groups.mmlu_redux import \
    mmlu_redux_summary_groups

SUMMARIZER_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        'opencompass',
        'configs',
        'summarizers',
        'mmlu_redux.py',
    ))


class TestMMLUReduxSummarizer:

    def test_groups_partition_all_subjects(self):
        groups = {
            group['name']: group['subsets']
            for group in mmlu_redux_summary_groups
        }
        all_subjects = groups.pop('mmlu_redux')

        assert len(all_subjects) == 57
        assert len(set(all_subjects)) == 57
        assert all(
            subject.startswith('mmlu_redux_') for subject in all_subjects)

        category_subjects = [
            subject for subsets in groups.values() for subject in subsets
        ]
        assert Counter(category_subjects) == Counter(all_subjects)
        assert len(groups) == 4

    def test_summarizer_config_lists_groups(self):
        config = Config.fromfile(SUMMARIZER_PATH)
        summarizer = config.summarizer

        assert summarizer['dataset_abbrs'] == [
            'mmlu_redux',
            'mmlu_redux-humanities',
            'mmlu_redux-stem',
            'mmlu_redux-social-science',
            'mmlu_redux-other',
        ]
        assert len(summarizer['summary_groups']) == 5
