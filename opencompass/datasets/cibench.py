import json
import os
import os.path as osp
import re
import subprocess
from collections import defaultdict
from typing import List, Optional

import numpy as np
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


def load_experiment(file: str) -> dict:
    """Load single experiment file with solutions."""
    with open(file, 'r') as f:
        notebook = json.load(f)
        example = notebook['cells']
        metadata = notebook['metadata']
        modules = metadata.get('modules', [])
        if modules:
            # these two annotations should be the same
            assert len(modules) == len(metadata.get('step_types'))
            # reformat annotations
            modules = [[_m.strip() for _m in _modules.split('&')]
                       for _modules in modules]
        questions = []
        source_codes = []
        outputs = []
        tags = []
        for cell in example:
            if cell['cell_type'] == 'markdown':
                text = ''.join(cell['source']).strip()
                if modules:
                    _modules = modules.pop(0)
                    text += f"Please use {' and '.join(_modules)} modules."
                text = text.strip() + '\n'
                # append the formatted text
                questions.append(text)
            elif cell['cell_type'] == 'code':
                source_codes.append(''.join(cell['source']))
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
                    tags.append('exec')
                    outputs.append(None)
    return dict(
        experiment=file,
        questions=sum(([
            dict(role='user', content=question),
            dict(role='assistant', content=source_code)
        ] for question, source_code in zip(questions, source_codes)), []),
        references=dict(outputs=outputs,
                        tags=tags,
                        metadata=metadata,
                        experiment=file),
    )


@LOAD_DATASET.register_module()
class CIBenchDataset(BaseDataset):
    """Code Interpreter dataset."""

    @staticmethod
    def load(path: str):
        """Load whole dataset."""
        assert os.path.exists(path), f'Path {path} does not exist.'
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
        text_evaluator (optional, dict): The text evaluator for text result
            comparison[]. Defaults to None, which use Rouge as defaults.
            Please notice that a extra key for `metric_name` should be set
            to get the exact metric result, such as `rouge1`.
        output_dir (optional, str): The directory to save experiment
            files in a markdown or notebook format.
        with_ipynb (bool): Generate ipynb correspondingly.
            Defaults to False.
        user_data_dir (str): The directory to load local files.
            Defaults to 'ENV', which means use environment variable
            `USER_DATA_DIR` to get the data dir.
    """

    def __init__(self,
                 text_evaluator: Optional[dict] = None,
                 output_dir: Optional[str] = None,
                 with_ipynb: bool = False,
                 user_data_dir: str = 'ENV') -> None:
        if text_evaluator is None:
            from opencompass.openicl.icl_evaluator import RougeEvaluator
            self.text_evaluator = ICL_EVALUATORS.build(
                dict(type=RougeEvaluator))
            self.text_eval_metric = 'rouge1'
        else:
            self.text_eval_metric = text_evaluator.pop('metric_name')
            self.text_evaluator = ICL_EVALUATORS.build(text_evaluator)
        # TODO: should use work dir for this task.
        self.output_dir = output_dir
        self.user_data_dir = self.check_user_data_dir(user_data_dir)
        self.with_ipynb = with_ipynb
        self.TAG_MAPPING = {
            'exec': ('executable', self.valid_step),
            'general': ('general_correct', self.correct_step),
            'num': ('numeric_correct', self.correct_step),
            'text': ('text_score', self.text_step),
            'vis': ('vis_sim', self.vis_similarity_step),
        }

    def check_user_data_dir(self, user_data_dir):
        if user_data_dir == 'ENV':
            user_data_dir = os.environ.get('USER_DATA_DIR', '')
        user_data_dir = user_data_dir.rstrip('/')
        basename = osp.basename(user_data_dir)
        if basename and basename != 'data':
            user_data_dir = osp.join(user_data_dir, 'data')
            assert osp.exists(user_data_dir), \
                f'a subfolder named `data` should exist under {user_data_dir}.'
        elif basename:
            assert osp.exists(user_data_dir), \
                f'{user_data_dir} does not exist.'
        return user_data_dir

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

    def text_step(self, step, target):
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
                            score = self.text_evaluator.score([out], [target])
                            return score[self.text_eval_metric] / 100
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
        p_list = []
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
            # The result might be different with infer process
            # please check carefully
            # convert markdown to ipynb and exectue with error tolerance
            if self.with_ipynb:
                p = subprocess.Popen(
                    'jupytext --to ipynb --pipe-fmt ipynb '
                    "--pipe 'jupyter nbconvert --to ipynb --execute "
                    f"--allow-errors --stdin --stdout' {md_file}",
                    shell=True)
                p_list.append(p)
        # TODO: async wait
        for p in p_list:
            p.wait()

    def set_data_dir(self, work_dir):
        """Set work directory and link data files for save notebook results."""
        if self.user_data_dir:
            basename = osp.basename(self.user_data_dir)

            if not osp.exists(osp.join(self.output_dir, basename)):
                os.symlink(self.user_data_dir,
                           osp.join(self.output_dir, basename))
        os.chdir(work_dir)

    def unset_data_dir(self, work_dir):
        """Change work directory and keep the symlink."""
        os.chdir(work_dir)

    def single_exp(self, gold, steps):
        tags = gold['tags']
        outputs = gold['outputs']
        metadata = gold['metadata']
        hard_tags = metadata.get('step_types', [])
        if hard_tags:
            tags = hard_tags

        # executable: exec succeed
        # general_correct: general correct
        # numeric_correct: numerical correct
        # text_score: text score
        # vis_sim: visual similarity
        result = defaultdict(list)
        for tag, step, output in zip(tags, steps, outputs):
            # check whether this step is valid
            result['executable'].append(self.valid_step(step))
            if tag != 'exec':
                key, func = self.TAG_MAPPING[tag]
                result[key].append(func(step, output))

        # add missing metric for better analyse if not exists
        if hard_tags:
            check_tags = ['exec', 'num', 'text', 'vis']
        else:
            check_tags = ['exec', 'general', 'vis']
        for tag in check_tags:
            key = self.TAG_MAPPING[tag][0]
            if key not in result:
                result[key] = []

        return result

    def get_output_dir(self):
        """Get output dir from eval task.

        Notice: output dir should be in format xxx/data.
        All the needed files should be
        """
        # hard hack for get output dir from eval task
        if hasattr(self, '_out_dir') and self.output_dir is None:
            self.output_dir = self._out_dir

    def score(self, predictions: List, references: List, steps: List,
              origin_prompt: List):
        """Calculate accuracy."""
        if len(steps) != len(references):
            return {'error': 'steps and refrs have different length'}
        cwd = os.getcwd()
        self.get_output_dir()
        if self.output_dir:
            if not osp.exists(self.output_dir):
                os.makedirs(self.output_dir)
            self.set_data_dir(self.output_dir)
            self.save_results(origin_prompt, steps)
            self.unset_data_dir(cwd)

        total_results = defaultdict(float)
        total_scores = defaultdict(float)
        total_nums = defaultdict(int)
        for gold, single_steps in zip(references, steps):
            result = self.single_exp(gold, single_steps)

            for k, v in result.items():
                total_scores[k] += sum(v)
                total_nums[k] += len(v)

        for k, v in total_scores.items():
            if total_nums[k] > 0:
                total_results[k] = total_scores[k] / total_nums[k] * 100
            else:
                total_results[k] = -1

        return total_results
