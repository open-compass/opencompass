import json
import os
import os.path as osp
import re
from typing import List, Optional

import numpy as np
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


def load_experiment(file: str) -> dict:
    """Load single experiment file with solutions."""
    with open(file, 'r') as f:
        notebook = json.load(f)
        example = notebook['cells']

        questions = []
        outputs = []
        tags = []
        for cell in example:
            if cell['cell_type'] == 'markdown':
                text = ''.join(cell['source'])
                # append the formatted text
                questions.append(text)
            elif cell['cell_type'] == 'code':
                if cell['outputs'] and 'data' in cell['outputs'][-1]:
                    if 'image/png' in cell['outputs'][-1]['data']:
                        # skip vis temporarily due to lack of evaluation
                        tags.append('vis')
                        outputs.append(
                            cell['outputs'][-1]['data']['image/png'])
                    elif 'text/plain' in cell['outputs'][-1]['data']:
                        tags.append('general')
                        outputs.append(''.join(
                            cell['outputs'][-1]['data']['text/plain']))
                else:
                    tags.append('executable')
                    outputs.append(None)
    return dict(
        experiment=file,
        questions=questions,
        references=dict(outputs=outputs, tags=tags, experiment=file),
    )


@LOAD_DATASET.register_module()
class CIBenchDataset(BaseDataset):
    """Code Interpreter dataset."""

    @staticmethod
    def load(path: str):
        """Load whole dataset."""
        data_list = []
        for cwd, dirs, files in os.walk(path):
            dirs.sort()
            files.sort()
            for f in files:
                if '.ipynb' in f:
                    try:
                        data = load_experiment(os.path.join(cwd, f))
                    except Exception:
                        print(f'Error with file {os.path.join(cwd, f)}')
                        continue
                    data_list.append(data)

        dataset = Dataset.from_list(data_list)
        return dataset


class CIBenchEvaluator(BaseEvaluator):
    """Evaluator for CI dataset.

    Args:
        output_dir (optional, str): The directory to save experiment
            files in a markdown or notebook format.
        user_data_dir (str): The directory to load local files.
            Defaults to 'ENV', which means use environment variable
            `USER_DATA_DIR` to get the data dir.
    """

    def __init__(self,
                 output_dir: Optional[str] = None,
                 user_data_dir: str = 'ENV') -> None:
        # TODO: should use work dir for this task.
        self.output_dir = output_dir
        if user_data_dir == 'ENV':
            user_data_dir = os.environ.get('USER_DATA_DIR', '')
        self.user_data_dir = user_data_dir

    @staticmethod
    def valid_step(step):
        """Whether the step is executable and valid."""
        # Found the latest code interpreter to determine valid
        for action in step[::-1]:
            if action['type'] == 'IPythonInterpreter':
                if action['errmsg']:
                    return False
                else:
                    return True
        # No code interpreter for this step, reckon as False
        return False

    @staticmethod
    def correct_step(step, target):
        """Whether the step output is correct."""
        # Found the latest code interpreter to determine correct
        for action in step[::-1]:
            if action['type'] == 'IPythonInterpreter':
                if action['result']:
                    try:
                        pred = action['result']['text']
                        match = re.search('```\n(.*?)\n```', pred, re.DOTALL)
                        if match:
                            out = match.group(1)
                            return out == target or out in target
                    except Exception:
                        return False
        # Fall back to False
        return False

    @staticmethod
    def vis_similarity_step(step, target):
        """Whether the step output image has the same structure similarity with
        the given images."""
        # Found the latest code interpreter to determine correct
        import base64

        import skimage

        for action in step[::-1]:
            if action['type'] == 'IPythonInterpreter':
                if action['result']:
                    try:
                        pred = action['result']['text']
                        match = re.search(r'!\[fig-[0-9]*\]\((.*?)\)', pred,
                                          re.DOTALL)
                        if match:
                            img_pred = match.group(1)
                        img2 = base64.b64decode(target)
                        img2 = skimage.io.imread(img2, plugin='imageio')
                        img1 = skimage.io.imread(img_pred, plugin='imageio')
                        img1 = skimage.transform.resize(img1, img2.shape[:2])
                        img1 = 255 * img1
                        # Convert to integer data type pixels.
                        img1 = img1.astype(np.uint8)
                        ssim = skimage.metrics.structural_similarity(
                            img1, img2, channel_axis=-1)
                        # mse = skimage.metrics.mean_squared_error(img1, img2)
                        # ssim greater better
                        # mse smaller better but has no upper bound
                        return ssim
                    except Exception:
                        return 0
        # Fall back to 0
        return 0

    def save_results(self, origin_prompt, steps):
        """Save the prediction result in a markdown and notebook format."""

        def check_jupytext():
            """Check requirements existence."""
            from shutil import which

            assert which('jupytext'), (
                "Please install jupytext use 'pip install jupytext' to ensure"
                'the conversion processes.')

        check_jupytext()
        from opencompass.lagent.actions.ipython_interpreter import extract_code
        for idx, (example_origin_prompt,
                  example_steps) in enumerate(zip(origin_prompt, steps)):
            markdown_lines = []
            for prompt, step in zip(example_origin_prompt, example_steps):
                for action in step[::-1]:
                    if action['type'] == 'IPythonInterpreter':
                        valid_action = action
                        break
                    # fall back to final action
                    valid_action = step[-1]
                markdown_lines.append(prompt)
                markdown_lines.append('\n')
                code_text = valid_action['args']['text']
                code_text = extract_code(code_text)
                code_text = '```python\n' + code_text + '\n```'
                markdown_lines.append(code_text)
                markdown_lines.append('\n')

            md_file = f'experiment{idx}.md'
            with open(md_file, 'w') as f:
                f.writelines(markdown_lines)

            # TODO: be careful for this
            # convert markdown to ipynb and exectue with error tolerance
            # subprocess.Popen(
            #     "jupytext --to ipynb --pipe-fmt ipynb "
            #     "--pipe 'jupyter nbconvert --to ipynb --execute "
            #     f"--allow-errors --stdin --stdout' {md_file}",
            #     shell=True)

    def set_data_dir(self, work_dir):
        """Set work directory and link data files for save notebook results."""
        if self.user_data_dir:
            if self.user_data_dir.endswith('/'):
                basename = osp.basename(osp.split(self.user_data_dir)[0])
            else:
                basename = osp.basename(self.user_data_dir)
            if not osp.exists(osp.join(self.output_dir, basename)):
                os.symlink(self.user_data_dir,
                           osp.join(self.output_dir, basename))
        os.chdir(work_dir)

    def unset_data_dir(self, work_dir):
        """Change work directory and keep the symlink."""
        os.chdir(work_dir)

    def score(self, predictions: List, references: List, steps: List,
              origin_prompt: List):
        """Calculate accuracy."""
        cwd = os.getcwd()
        if self.output_dir:
            if not osp.exists(self.output_dir):
                os.makedirs(self.output_dir)
            self.set_data_dir(self.output_dir)
            self.save_results(origin_prompt, steps)
            self.unset_data_dir(cwd)

        num_cells_list = []
        num_general_list = []
        passed_list = []
        correct_list = []
        vis_list = []
        for gold, single_steps in zip(references, steps):
            tags = gold['tags']
            outputs = gold['outputs']
            num_cells = len(tags)
            num_general = sum([tag == 'general' for tag in tags])

            passed = sum([self.valid_step(step) for step in single_steps])
            correct = 0
            vis_sim = []
            for tag, step, output in zip(tags, single_steps, outputs):
                if tag == 'general':
                    correct += self.correct_step(step, output)
                elif tag == 'vis':
                    vis_sim.append(self.vis_similarity_step(step, output))

            num_cells_list.append(num_cells)
            num_general_list.append(num_general)
            passed_list.append(passed)
            correct_list.append(correct)
            if vis_sim:
                vis_list.append(sum(vis_sim) / len(vis_sim))
            else:
                vis_list.append(-1)

        if len([v for v in vis_list if v >= 0]) > 0:
            visualize_similarity = sum([v for v in vis_list if v >= 0]) / len(
                [v for v in vis_list if v >= 0])
        else:
            # not valid
            visualize_similarity = -1

        if sum(num_general_list) > 0:
            general_accuracy = sum(correct_list) / sum(num_general_list)
        else:
            # not valid
            general_accuracy = -1

        result = dict(
            executable_rate=sum(passed_list) / sum(num_cells_list) * 100,
            general_accuracy=general_accuracy * 100,
            visualize_similarity=visualize_similarity * 100,
            num_cells_list=num_cells_list,
            num_general_list=num_general_list,
            passed_list=passed_list,
            correct_list=correct_list,
            vis_list=vis_list,
        )
        return result
