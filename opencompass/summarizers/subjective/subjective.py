# flake8: noqa: E501
from datetime import datetime

from mmengine import ConfigDict


class SubjectiveSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, function: str) -> None:
        self.cfg = config
        self.function = function

    def summarize(self,
                  time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            None
        """
        # This is a placeholder for the summarizer. If you want to understand the logic of the summarizer in the subjective evaluation, please refer to the visualization section of opencompass/cli/main.py.
        return None
