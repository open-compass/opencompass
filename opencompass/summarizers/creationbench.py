# flake8: noqa: E501
import csv
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
from mmengine import ConfigDict

try:
    from prettytable import from_csv
except ImportError:
    from_csv = None

from opencompass.utils import model_abbr_from_cfg

from .alignmentbench import AlignmentBenchSummarizer, post_process_alignbench
from .subjective_post_process import post_process_autoj, post_process_judgelm
from .utils import get_judgeanswer_and_reference, get_outdir

CATEGORIES = {
    '中文': ['内容扩写_ZH', '内容续写_ZH', '内容改写_ZH'],
    '英文': ['内容扩写_EN', '内容续写_EN', '内容改写_EN'],
}

All_Dimensions = [
    'Creativity', 'Richness', 'User Demand Fulfillment', 'Logical Coherence',
    'Overall Score', '创造性', '丰富度', '满足用户需求', '逻辑连贯性', '综合得分'
]


def post_process_creationbench(judgement: str,
                               all_dimensions=All_Dimensions,
                               possible_keys=['综合得分', 'Overall Score']):
    """Input a string like below:

    xxx{'事实正确性': 1, '满足用户需求': 1, '清晰度': 2, '完备性': 1, '综合得分': 1}xxx,
    and extract each score
    """
    return post_process_alignbench(judgement, all_dimensions, possible_keys)


class CreationBenchSummarizer(AlignmentBenchSummarizer):
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, judge_type: str) -> None:
        super().__init__(config, judge_type)
        self.judge_map = {
            'general': post_process_creationbench,
            'autoj': post_process_autoj,
            'judgelm': post_process_judgelm
        }
        self.judge_function = self.judge_map[self.judge_type]
        self.category = CATEGORIES

    def summarize(self,
                  time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """
        super().summarize(time_str)
