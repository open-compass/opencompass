import json
import os
import os.path as osp
import re
import subprocess
from collections import defaultdict
from inspect import signature
from typing import List, Optional

import numpy as np
from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.datasets.gsm8k import gsm8k_postprocess
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET


def load_experiment(file: str) -> dict:
    """Load single experiment file with solutions for template experiment."""
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
        thoughts = []
        outputs = []
        tags = []

        for cell in example:
            if cell['cell_type'] == 'markdown':
                text = ''.join(cell['source']).strip()
                try:
                    text, thought = text.split('\n\nThought: ')
                except ValueError:
                    thought = ' '
                if modules:
                    _modules = modules.pop(0)
                    if 'chinese' not in file:
                        text += f"Please use {' and '.join(_modules)} modules."
                    else:
                        text += f"请用 {' 和 '.join(_modules)} 模块."
                text = text.strip() + '\n'
                # append the formatted text
                questions.append(text)
                thoughts.append(thought)
            elif cell['cell_type'] == 'code':
                source_codes.append(''.join(cell['source']))
                output_flag = False
                if cell['outputs']:
                    for _output in cell['outputs']:
                        if _output['output_type'] == 'display_data':
                            assert not output_flag
                            if 'image/png' in _output['data']:
                                output_flag = True
                                tags.append('vis')
                                outputs.append(_output['data']['image/png'])
                    for _output in cell['outputs'][::-1]:
                        if output_flag:
                            break
                        if _output['output_type'] == 'stream' and _output[
                                'name'] == 'stdout':
                            assert not output_flag
                            output_flag = True
                            tags.append('general')
                            outputs.append(''.join(_output['text']))
                        elif _output['output_type'] == 'execute_result':
                            assert not output_flag
                            output_flag = True
                            tags.append('general')
                            outputs.append(''.join(
                                _output['data']['text/plain']))
                if not output_flag:
                    # no output fallback to exec
                    tags.append('exec')
                    outputs.append(None)
    return dict(
        experiment=file,
        questions=sum(([
            dict(role='user', content=question),
            dict(role='assistant', content=thought + '**split**' + source_code)
        ]
                       for question, source_code, thought in zip(
                           questions, source_codes, thoughts)), []),
        references=dict(outputs=outputs,
                        tags=tags,
                        metadata=metadata,
                        experiment=file),
    )


def check_internet():
    """A tricky way to check internet."""
    import socket

    import nltk
    socket.setdefaulttimeout(10)
    ret = nltk.download('stopwords', quiet=True)
    socket.setdefaulttimeout(None)
    if not ret:
        raise ConnectionError('CIBench needs internet to get response. Please'
                              'check your internet and proxy.')


@LOAD_DATASET.register_module()
class CIBenchDataset(BaseDataset):
    """Code Interpreter dataset for template dataset."""

    @staticmethod
    def load(path: str, internet_check: bool = False):
        """Load whole dataset.

        Args:
            path(str): Path of cibench dataset.
            internet_check(bool): Whether to check internet.
                Defaults to False.
        """
        if internet_check:
            check_internet()
        assert os.path.exists(path), f'Path {path} does not exist.'
        data_list = []
        for cwd, dirs, files in os.walk(path):
            dirs.sort()
            files.sort()
            for f in files:
                if '.ipynb' in f:
                    data = load_experiment(os.path.join(cwd, f))
                    data_list.append(data)

        dataset = Dataset.from_list(data_list)
        return dataset


def sklearn_ssim(pred_img, target_img):
    import base64

    import skimage
    img2 = base64.b64decode(target_img)
    img2 = skimage.io.imread(img2, plugin='imageio')
    img1 = skimage.io.imread(pred_img, plugin='imageio')
    img1 = skimage.transform.resize(img1, img2.shape[:2])
    img1 = 255 * img1
    # Convert to integer data type pixels.
    img1 = img1.astype(np.uint8)
    ssim = skimage.metrics.structural_similarity(img1, img2, channel_axis=-1)
    return ssim


JUDGE_PROMPT_CN = """你是一个擅长评价可视化能力的助手。
请你以公正的评判者的身份，评估一个AI模型对可视化相关问题生成的代码所绘制图像的质量。
我们会给您提供一个代码可视化问题，和需要你评估的AI模型生成的代码所绘制的图像。当你开始你的评估时，你需要遵守以下的流程：
1. 针对图像，给可视化能力一个1～10的分数，仅需返回数字，无需任何其他描述。
2. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。

当图像完全无法反映出所给定的指令内容时，此类评分得到1到2分。
当图像能够部分体现出所给定的指令内容，但在具体的细节表达上有很大的缺失时，此类评分为3到4分。
当图像基本能够符合所给定的指令，但是在图像的美观性上呈现一般，没有特别出彩的地方时，此类评分可以得到5到6分。
当图像能够较好地匹配上所给的指令，并且在图像的美观性上有所表现，如在颜色搭配、形状设计等方面有一些新意时，此类评分可以得到7到8分。
当图像完全匹配上所给的指令，涵盖了指令中的所有细节，并且在图像的美观性上表现出色，此类评分才能得到9到10分。

[可视化问题]：{question}
"""  # noqa

JUDGE_PROMPT = """You are an assistant skilled in assessing visualization capabilities.
In the capacity of a fair judge, you will evaluate the quality of images drawn by an AI model generating code for visualization-related problems. We will provide you with a code visualization problem and an image drawn by the code created by the AI model you need to assess. When you start your assessment, you must adhere to the following process:
1. Rate the visualization capability with a score between 1 and 10 for the image, returning only the number without any additional descriptions.
2. Your scoring needs to be as rigorous as possible, and it should follow the scoring rules below: Overall, the higher the quality of the model's response, the higher the score.

A score of 1 to 2 is given when the image cannot reflect the given instruction content at all.
A score of 3 to 4 is given when the image can partly reflect the given instruction content, but there is a significant lack of specific detail expression.
If the image basically meets the given instructions, but the aesthetic quality of the image is average without any outstanding features, this kind of rating can get a score of 5 to 6.
When the image matches the given instructions well, and shows some aesthetic appeal, such as some originality in color matching and shape design, this kind of rating can get a score of 7 to 8.
Only when the image completely matches the given instructions, covers all the details in the instructions, and performs excellently in terms of aesthetics, can this kind of rating get a score of 9 to 10.

[Visualization Problem]:{question}
"""  # noqa


def vl_model_score(model, pred_img, ori_prompt, judge_prompt):
    response = model.interleave_generate(
        [judge_prompt.format(question=ori_prompt), pred_img])
    score = gsm8k_postprocess(response)
    try:
        score = int(float(score))
        assert score <= 10 and score >= 1
        return score / 10
    except Exception as e:
        raise ValueError(f'Evaluation failed {e}. Check log for details.')


@ICL_EVALUATORS.register_module()
class CIBenchEvaluator(BaseEvaluator):
    """Evaluator for CI dataset.

    Args:
        text_evaluator (optional, dict): The text evaluator for text result
            comparison[]. Defaults to None, which use rouge as defaults.
            Please notice that a extra key for `metric_name` should be set
            to get the exact metric result, such as `rouge1`.
        vis_evaluator (optional, dict): The vis evaluator for visualization
            score. Defaults to None, which means use skimage. Otherwise
            provide dict from VLMEvalKit.
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
                 vis_evaluator: Optional[dict] = None,
                 output_dir: Optional[str] = None,
                 with_ipynb: bool = False,
                 lang: str = 'en',
                 user_data_dir: str = 'ENV') -> None:
        # build text evaluator
        if text_evaluator is None:
            from opencompass.openicl.icl_evaluator import RougeEvaluator
            self.text_evaluator = ICL_EVALUATORS.build(
                dict(type=RougeEvaluator))
            self.text_eval_metric = 'rouge1'
        else:
            self.text_eval_metric = text_evaluator.pop('metric_name')
            self.text_evaluator = ICL_EVALUATORS.build(text_evaluator)
        # build visual evaluator
        if vis_evaluator is None:
            self.vis_evaluator = None
        else:
            try:
                from vlmeval.config import supported_VLM
            except ImportError as e:
                raise ImportError(
                    f'{e}. Please install vlmeval following: https://github.com/open-compass/VLMEvalKit'  # noqa
                )
            assert vis_evaluator['type'] in supported_VLM, ''
            self.vis_evaluator = supported_VLM[vis_evaluator.pop('type')](
                **vis_evaluator)

        assert lang in ['en', 'cn'], 'Only `en` and `cn` are supported.'
        self.lang = lang
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
            default_path = osp.abspath('./data/cibench_dataset/datasources')
            user_data_dir = os.environ.get('USER_DATA_DIR', default_path)
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
                    return True, False
                else:
                    return True, True
        # No code interpreter for this step, reckon as False
        return False, False

    @staticmethod
    def correct_step(step, target) -> dict:
        """Whether the step output is correct."""
        # Found the latest code interpreter to determine correct
        for action in step[::-1]:
            if action['type'] == 'IPythonInterpreter':
                if action['result']:
                    try:
                        pred = action['result']['text']
                        match_exec = re.search(
                            'execute_result:\n\n```\n(.*?)\n```', pred,
                            re.DOTALL)
                        match_stdout = re.search('stdout:\n\n```\n(.*?)\n```',
                                                 pred, re.DOTALL)
                        # get pred result from execute_result by default
                        # else stdout
                        if match_exec and match_stdout:
                            match = match_exec
                        elif match_exec:
                            match = match_exec
                        elif match_stdout:
                            match = match_stdout
                        else:
                            match = None
                        if match:
                            out = match.group(1)
                            score = (out.strip() == target.strip()
                                     or target.strip() in out.strip())
                            return {'score': score, 'gt': target, 'pred': out}
                    except Exception:
                        return {'score': 0, 'gt': target}
        # Fall back to False
        return {'score': 0, 'gt': target}

    def text_step(self, step, target) -> dict:
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
                            score = score[self.text_eval_metric] / 100
                            return {
                                'score': score,
                                'gt_text': target,
                                'pred_text': out
                            }
                    except Exception:
                        return {'score': 0, 'gt_text': target}
        # Fall back to False
        return {'score': 0, 'gt_text': target}

    def vis_similarity_step(self, step, target, ori_prompt) -> dict:
        """Whether the step output image has the same structure similarity with
        the given images."""
        # Found the latest code interpreter to determine correct
        for action in step[::-1]:
            if action['type'] == 'IPythonInterpreter':
                if action['result']:
                    try:
                        pred = action['result']['image_path']
                        match = re.search(r'!\[fig-[0-9]*\]\((.*?)\)', pred,
                                          re.DOTALL)
                        if match:
                            img_pred = match.group(1)
                        if self.vis_evaluator is None:
                            # ssim greater better
                            score = sklearn_ssim(img_pred, target)
                            return {'score': score, 'pred_img': img_pred}
                        else:
                            # TODO: the following code will be removed later.
                            if self.lang == 'cn':
                                score = vl_model_score(self.vis_evaluator,
                                                       img_pred, ori_prompt,
                                                       JUDGE_PROMPT_CN)
                                return {'score': score, 'pred_img': img_pred}
                            elif self.lang == 'en':
                                score = vl_model_score(self.vis_evaluator,
                                                       img_pred, ori_prompt,
                                                       JUDGE_PROMPT)
                                return {'score': score, 'pred_img': img_pred}
                    except Exception:
                        return {'score': 0}
        # Fall back to 0
        return {'score': 0}

    def save_results(self, origin_prompt, steps, references):
        """Save the prediction result in a markdown and notebook format."""

        from opencompass.lagent.actions.ipython_interpreter import extract_code

        def check_jupytext():
            """Check requirements existence."""
            from shutil import which

            assert which('jupytext'), (
                "Please install jupytext use 'pip install jupytext' to ensure"
                'the conversion processes.')

        check_jupytext()
        p_list = []
        total_results = defaultdict(float)
        total_scores = defaultdict(float)
        total_nums = defaultdict(int)

        for idx, (example_origin_prompt, example_steps,
                  gold) in enumerate(zip(origin_prompt, steps, references)):
            # get result count
            result, exp_output = self.single_exp(gold, example_steps,
                                                 example_origin_prompt)
            for k, v in result.items():
                total_scores[k] += sum(v)
                total_nums[k] += len(v)

            markdown_lines = []
            for prompt, step, step_output in zip(example_origin_prompt,
                                                 example_steps, exp_output):
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
                markdown_lines.append('\n'.join(
                    [f'{k}: {v}' for k, v in step_output.items()]))
                markdown_lines.append('\n\n')

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

        # get final scores
        for k, v in total_scores.items():
            if total_nums[k] > 0:
                total_results[k] = total_scores[k] / total_nums[k] * 100
            else:
                total_results[k] = -1
        return total_results

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

    def single_exp(self, gold, steps, single_ori_prompt):
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

        # create empty results
        result = dict()
        if hard_tags:
            check_tags = ['exec', 'num', 'text', 'vis']
        else:
            check_tags = ['exec', 'general', 'vis']
        for tag in check_tags:
            key = self.TAG_MAPPING[tag][0]
            result[key] = []
        result['tool_rate'] = []

        exp_output = []
        for tag, step, output, ori_prompt in zip(tags, steps, outputs,
                                                 single_ori_prompt):
            # check whether this step is valid
            tool_correct, exec_correct = self.valid_step(step)
            result['tool_rate'].append(tool_correct)
            result['executable'].append(exec_correct)
            eval_output = {}
            if tag != 'exec':
                key, func = self.TAG_MAPPING[tag]
                kwargs = dict(step=step, target=output, ori_prompt=ori_prompt)
                kwargs = {k: kwargs[k] for k in signature(func).parameters}
                eval_output = func(**kwargs)
                result[key].append(eval_output['score'])
            exp_output.append(eval_output)

        return result, exp_output

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
            total_results = self.save_results(origin_prompt, steps, references)
            self.unset_data_dir(cwd)

        return total_results
