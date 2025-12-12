# opencompass/datasets/opi/evaluator.py

import json
import re

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .utils.metrics4all import calculate_metrics, calculate_rouge_l


@LOAD_DATASET.register_module()
class opiDataset(BaseDataset):

    @staticmethod
    def load(train_path, test_path, max_cut=-1, mini_set=False, hf_hub=False):
        if (hf_hub is True):
            # load from huggingface hub
            train_data = []
            repo_id = test_path.split('/')[0] + '/' + test_path.split('/')[1]
            train_path = train_path.split(repo_id + '/')[1]
            test_path = test_path.split(repo_id + '/')[1]

            train_path = hf_hub_download(repo_id,
                                         train_path,
                                         repo_type='dataset')
            test_path = hf_hub_download(repo_id,
                                        test_path,
                                        repo_type='dataset')

        # load from local json file
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        # train_data = train_data[:10]
        # # Limit the dataset to 10 samples for testing purposes
        # test_data = test_data[:10]
        if mini_set:
            import random
            random.seed(1024)
            test_data = random.sample(test_data, 150)
            random.seed()

        dataset = DatasetDict({
            'train': Dataset.from_list(train_data),
            'test': Dataset.from_list(test_data)
        })
        return dataset


def extract_answer_part(outputs, left_tag, right_tag, mode='tag'):
    assert mode in ('tag', 'direct')
    assert isinstance(outputs, list)

    answers = []
    for text in outputs:
        if mode == 'direct' or (left_tag is None and right_tag is None):
            text = text.replace('<unk>', '').replace('</s>', '').strip()
            answers.append(text.strip())
            continue

        left_tag_pos = text.find(left_tag)
        if left_tag_pos == -1:
            answers.append('')
            continue
        right_tag_pos = text.find(right_tag)
        if right_tag_pos == -1:
            answers.append('')
            continue
        text = text[left_tag_pos + len(left_tag):right_tag_pos].strip()
        answers.append(text)
    return answers


@TEXT_POSTPROCESSORS.register_module('opi_postprocess')
def opi_postprocess(text, task, *args, **kwargs):
    print(f'task: {task}, text: {text}')
    text = text.strip()
    text = re.sub(r'<\|endoftext\|>', '', text)
    text = re.sub(r'<\|im_end\|>', '', text)
    return text


class opi_Evaluator(BaseEvaluator):

    def __init__(self, task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        if not isinstance(predictions[0], list):
            predictions = [[pred] for pred in predictions]
        if not isinstance(references[0], list):
            references = [[ref] for ref in references]

        if self.task == 'Function':
            return self._evaluate_function(predictions, references)
        elif self.task == 'Subcellular_localization':
            return self._evaluate_subcellular_localization(
                predictions, references)
        elif self.task == 'Fold_type':
            return self._evaluate_fold_type(predictions, references)
        elif self.task in ('EC_number', 'GO', 'Keywords', 'gSymbol2Tissue',
                           'gSymbol2Cancer', 'gName2Cancer'):
            return self._evaluate_multilabel(predictions, references)
        else:
            return self._evaluate_general(predictions, references)

    def _evaluate_function(self, predictions, references):
        """评估功能描述任务，使用 ROUGE-L"""
        # if not METRICS_AVAILABLE:
        #     return self._evaluate_text_similarity(predictions, references)

        rouge_ls = []
        for pred_list, ref_list in zip(predictions, references):
            pred = pred_list[0].strip()
            ref = ref_list[0].strip()

            # 确保输出和目标是列表格式
            if isinstance(pred, str):
                pred = [pred]
            if isinstance(ref, str):
                ref = [ref]

            rouge_l = calculate_rouge_l(pred, ref)
            rouge_ls.append(rouge_l)

        mean_rouge_l = sum(rouge_ls) / len(rouge_ls) if rouge_ls else 0
        return {
            'ROUGE-L': round(mean_rouge_l, 4),
            # 'total': len(predictions)
        }

    def _evaluate_subcellular_localization(self, predictions, references):
        """评估亚细胞定位任务，使用准确率"""
        # if not METRICS_AVAILABLE:
        #     return self._evaluate_general(predictions, references)

        accuracies = []
        for pred_list, ref_list in zip(predictions, references):
            pred = pred_list[0].strip()
            ref = ref_list[0].strip()

            # 确保输出和目标是列表格式
            if isinstance(pred, str):
                pred = [pred]
            if isinstance(ref, str):
                ref = [ref]

            accuracy, _, _, _ = calculate_metrics(pred, ref)
            accuracies.append(accuracy)

        mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        return {
            'Accuracy': round(mean_accuracy, 4),
            # 'total': len(predictions)
        }

    def _evaluate_fold_type(self, predictions, references):
        """评估折叠类型任务，使用与 accuracy4fold_type.py 相同的计算方式"""
        # 初始化计数器
        correct_predictions = 0
        total_predictions = 0

        # 评估每个预测结果
        for pred_list, ref_list in zip(predictions, references):
            pred = pred_list[0].strip()
            ref = ref_list[0].strip()

            # 直接比较预测值和真实值
            if pred == ref:
                correct_predictions += 1
            total_predictions += 1

        # 计算准确率
        accuracy = correct_predictions / total_predictions \
            if total_predictions > 0 else 0

        return {
            'Accuracy': round(accuracy, 4),
            # 'correct': correct_predictions,
            # 'total': total_predictions
        }

    def _evaluate_multilabel(self, predictions, references):
        """评估多标签任务（EC_number, GO, Keywords）"""
        # if not METRICS_AVAILABLE:
        #     return self._evaluate_general(predictions, references)

        precisions = []
        recalls = []
        f1_scores = []

        for pred_list, ref_list in zip(predictions, references):
            pred = pred_list[0].strip()
            ref = ref_list[0].strip()

            # if isinstance(pred, str):
            #     pred = re.split(r'[;,，；]\s*', pred)
            # if isinstance(ref, str):
            #     ref = re.split(r'[;,，；]\s*', ref)
            if isinstance(pred, str):
                pred = [
                    p.strip() for p in re.split(r'[;,，；]\s*', pred)
                    if p.strip()
                ]
            if isinstance(ref, str):
                ref = [
                    r.strip() for r in re.split(r'[;,，；]\s*', ref)
                    if r.strip()
                ]

            # 过滤空字符串
            # pred = [p for p in pred if p.strip()]
            # ref = [r for r in ref if r.strip()]
            # import pdb; pdb.set_trace()
            if ref:  # 只有当参考标签不为空时才计算
                _, precision, recall, f1 = calculate_metrics(pred, ref)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)

        mean_precision = sum(precisions) / len(precisions) if precisions else 0
        mean_recall = sum(recalls) / len(recalls) if recalls else 0
        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

        return {
            'Precision': round(mean_precision, 4),
            'Recall': round(mean_recall, 4),
            'F1 Score': round(mean_f1, 4),
            # 'total': len(predictions)
        }

    def _evaluate_text_similarity(self, predictions, references):
        """简单的文本相似度评估（当 ROUGE 不可用时）"""
        correct = 0
        total = len(predictions)

        for pred_list, ref_list in zip(predictions, references):
            pred = pred_list[0].lower().strip()
            ref = ref_list[0].lower().strip()

            # 简单的包含关系检查
            if pred == ref or pred in ref or ref in pred:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        return {
            'Text_Similarity': round(accuracy, 4),
            # 'correct': correct,
            # 'total': total
        }

    def _evaluate_general(self, predictions, references):
        """通用评估方法"""
        correct = 0
        total = len(predictions)

        for pred_list, ref_list in zip(predictions, references):
            pred = pred_list[0].lower().strip()
            ref = ref_list[0].lower().strip()

            if pred == ref:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        return {
            'Accuracy': round(accuracy, 4),
            # 'correct': correct,
            # 'total': total
        }
