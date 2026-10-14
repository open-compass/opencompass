# flake8: noqa
import json
import os
import re
from collections import defaultdict

from .icl_base_evaluator import BaseEvaluator


class JudgeEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        correct = 0
        count = 0
        details = []
        for prediction, reference in zip(predictions, references):
            choice = prediction.split("\"Choice\": \"Model ")[-1][0] if len(
                prediction) != 0 else None
            gold_winner = reference.get('winner', '')
            detail = {
                'pred': prediction,
                'answer': gold_winner,
                'correct': False
            }
            count += 1
            if choice == gold_winner:
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result


class RMBEvaluator(BaseEvaluator):

    def calculate_pair_accuracy(self, data):
        correct = 0
        total = 0
        for item in data:
            choice = item['choice']
            gold_winner = item['gold_winner']
            if choice and gold_winner:
                total += 1
                if gold_winner == choice:
                    correct += 1

        return correct / total if total > 0 else 0

    def calculate_bon_accuracy(self, data):
        bon_groups = defaultdict(list)

        for item in data:
            bon_uid = item['bon_uid']
            if bon_uid:
                choice = item['choice']
                gold_winner = item['gold_winner']
                if choice and gold_winner:
                    bon_groups[bon_uid].append(gold_winner == choice)

        correct_bons = 0
        for bon_uid, matches in bon_groups.items():
            if all(matches):
                correct_bons += 1

        return correct_bons / len(bon_groups) if bon_groups else 0

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        bon_help_list = []
        bon_harm_list = []
        pair_help_list = []
        pair_harm_list = []

        for prediction, reference in zip(predictions, references):
            choice = prediction.split("\"Choice\": \"Model ")[-1][0] if len(
                prediction) != 0 else None
            gold_winner = reference.get('winner', '')
            subset = reference.get('subset', '')
            goal = reference.get('goal', '')

            data_item = {
                'choice': choice,
                'gold_winner': gold_winner,
                'bon_uid': reference.get('bon_uid', ''),
                'pair_uid': reference.get('pair_uid', ''),
            }

            if subset == 'bon':
                if goal == 'Helpfulness':
                    bon_help_list.append(data_item)
                elif goal == 'Harmlessness':
                    bon_harm_list.append(data_item)
            elif subset == 'pair':
                if goal == 'Helpfulness':
                    pair_help_list.append(data_item)
                elif goal == 'Harmlessness':
                    pair_harm_list.append(data_item)

        bon_help_acc = self.calculate_bon_accuracy(
            bon_help_list) if bon_help_list else 0
        bon_harm_acc = self.calculate_bon_accuracy(
            bon_harm_list) if bon_harm_list else 0
        pair_help_acc = self.calculate_pair_accuracy(
            pair_help_list) if pair_help_list else 0
        pair_harm_acc = self.calculate_pair_accuracy(
            pair_harm_list) if pair_harm_list else 0

        result = {
            'bon_helpfulness_accuracy':
            bon_help_acc * 100,
            'bon_harmlessness_accuracy':
            bon_harm_acc * 100,
            'pair_helpfulness_accuracy':
            pair_help_acc * 100,
            'pair_harmlessness_accuracy':
            pair_harm_acc * 100,
            'bon_average': ((bon_help_acc + bon_harm_acc) / 2) * 100,
            'pair_average': ((pair_help_acc + pair_harm_acc) / 2) * 100,
            'total_accuracy':
            ((bon_help_acc + bon_harm_acc + pair_help_acc + pair_harm_acc) / 4)
            * 100
        }

        return result


R1_Score_MAP = {
    'Knowledge': {
        'Qwen2.5-32B-Instruct': 55,
        'Llama-3.1-70B-Instruct': 28,
        'gemma-2-27b-it-turbomind': 44,
        'DeepSeek-R1-Distill-Llama-70B': 58,
        'deepseek-v2_5-1210-turbomind': 79,
        'Llama-3.3-70B-Instruct': 46,
        'nvidia-Llama-3.1-Nemotron-70B-Instruct-HF': 76,
        'DeepSeek-R1-Distill-Qwen-32B': 56,
        'mixtral-large-instruct-2407-lmdeploy': 72,
        'Qwen2.5-72B-Instruct': 80
    },
    'Longtext': {
        'Qwen2.5-32B-Instruct': 45,
        'Llama-3.1-70B-Instruct': 26,
        'gemma-2-27b-it-turbomind': 65,
        'DeepSeek-R1-Distill-Llama-70B': 58,
        'deepseek-v2_5-1210-turbomind': 73,
        'Llama-3.3-70B-Instruct': 37,
        'nvidia-Llama-3.1-Nemotron-70B-Instruct-HF': 54,
        'DeepSeek-R1-Distill-Qwen-32B': 52,
        'mixtral-large-instruct-2407-lmdeploy': 63,
        'Qwen2.5-72B-Instruct': 77
    },
    'Reason_and_analysis': {
        'Qwen2.5-32B-Instruct': 60,
        'Llama-3.1-70B-Instruct': 23,
        'gemma-2-27b-it-turbomind': 46,
        'DeepSeek-R1-Distill-Llama-70B': 63,
        'deepseek-v2_5-1210-turbomind': 85,
        'Llama-3.3-70B-Instruct': 45,
        'nvidia-Llama-3.1-Nemotron-70B-Instruct-HF': 68,
        'DeepSeek-R1-Distill-Qwen-32B': 66,
        'mixtral-large-instruct-2407-lmdeploy': 56,
        'Qwen2.5-72B-Instruct': 78
    },
    'safe': {
        'Qwen2.5-32B-Instruct': 72,
        'Llama-3.1-70B-Instruct': 55,
        'gemma-2-27b-it-turbomind': 72,
        'DeepSeek-R1-Distill-Llama-70B': 55,
        'deepseek-v2_5-1210-turbomind': 72,
        'Llama-3.3-70B-Instruct': 64,
        'nvidia-Llama-3.1-Nemotron-70B-Instruct-HF': 76,
        'DeepSeek-R1-Distill-Qwen-32B': 55,
        'mixtral-large-instruct-2407-lmdeploy': 69,
        'Qwen2.5-72B-Instruct': 83
    },
    'Hallucination': {
        'Qwen2.5-32B-Instruct': 78,
        'Llama-3.1-70B-Instruct': 50,
        'gemma-2-27b-it-turbomind': 65,
        'DeepSeek-R1-Distill-Llama-70B': 61,
        'deepseek-v2_5-1210-turbomind': 66,
        'Llama-3.3-70B-Instruct': 48,
        'nvidia-Llama-3.1-Nemotron-70B-Instruct-HF': 75,
        'DeepSeek-R1-Distill-Qwen-32B': 60,
        'mixtral-large-instruct-2407-lmdeploy': 76,
        'Qwen2.5-72B-Instruct': 74
    },
    'chatQA': {
        'Qwen2.5-32B-Instruct': 39,
        'Llama-3.1-70B-Instruct': 25,
        'gemma-2-27b-it-turbomind': 56,
        'DeepSeek-R1-Distill-Llama-70B': 53,
        'deepseek-v2_5-1210-turbomind': 70,
        'Llama-3.3-70B-Instruct': 34,
        'nvidia-Llama-3.1-Nemotron-70B-Instruct-HF': 69,
        'DeepSeek-R1-Distill-Qwen-32B': 48,
        'mixtral-large-instruct-2407-lmdeploy': 55,
        'Qwen2.5-72B-Instruct': 68
    },
    'IF': {
        'Qwen2.5-32B-Instruct': 34,
        'Llama-3.1-70B-Instruct': 35,
        'gemma-2-27b-it-turbomind': 38,
        'DeepSeek-R1-Distill-Llama-70B': 50,
        'deepseek-v2_5-1210-turbomind': 63,
        'Llama-3.3-70B-Instruct': 37,
        'nvidia-Llama-3.1-Nemotron-70B-Instruct-HF': 62,
        'DeepSeek-R1-Distill-Qwen-32B': 41,
        'mixtral-large-instruct-2407-lmdeploy': 47,
        'Qwen2.5-72B-Instruct': 48
    },
    'LanTask': {
        'Qwen2.5-32B-Instruct': 62,
        'Llama-3.1-70B-Instruct': 29,
        'gemma-2-27b-it-turbomind': 53,
        'DeepSeek-R1-Distill-Llama-70B': 60,
        'deepseek-v2_5-1210-turbomind': 75,
        'Llama-3.3-70B-Instruct': 46,
        'nvidia-Llama-3.1-Nemotron-70B-Instruct-HF': 69,
        'DeepSeek-R1-Distill-Qwen-32B': 71,
        'mixtral-large-instruct-2407-lmdeploy': 48,
        'Qwen2.5-72B-Instruct': 74
    },
    'Creation': {
        'Qwen2.5-32B-Instruct': 40,
        'Llama-3.1-70B-Instruct': 34,
        'gemma-2-27b-it-turbomind': 55,
        'DeepSeek-R1-Distill-Llama-70B': 66,
        'deepseek-v2_5-1210-turbomind': 73,
        'Llama-3.3-70B-Instruct': 36,
        'nvidia-Llama-3.1-Nemotron-70B-Instruct-HF': 73,
        'DeepSeek-R1-Distill-Qwen-32B': 64,
        'mixtral-large-instruct-2407-lmdeploy': 43,
        'Qwen2.5-72B-Instruct': 67
    },
    'Code_and_AI': {
        'Qwen2.5-32B-Instruct': 44,
        'Llama-3.1-70B-Instruct': 32,
        'gemma-2-27b-it-turbomind': 34,
        'DeepSeek-R1-Distill-Llama-70B': 56,
        'deepseek-v2_5-1210-turbomind': 64,
        'Llama-3.3-70B-Instruct': 43,
        'nvidia-Llama-3.1-Nemotron-70B-Instruct-HF': 62,
        'DeepSeek-R1-Distill-Qwen-32B': 43,
        'mixtral-large-instruct-2407-lmdeploy': 51,
        'Qwen2.5-72B-Instruct': 60
    }
}


class Judgerbenchv2Evaluator(BaseEvaluator):

    def get_rank_dict(self, score_dict):
        sorted_models = sorted(score_dict.items(), key=lambda x: (-x[1], x[0]))
        return {
            model: rank + 1
            for rank, (model, _) in enumerate(sorted_models)
        }

    def extract_winner(self, s, lan):
        pattern = (r'"?(胜者)"?\s*:\s*"([A-Z])"' if lan.lower() in ['zh', 'cn']
                   else r'"?(winner)"?\s*:\s*"([A-Z])"')

        matches = re.findall(pattern, s)

        return matches[-1][1] if matches else None

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        correct = 0
        count = 0
        details = []
        Model_dict = {}
        for prediction, reference in zip(predictions, references):
            # pre-defines
            ModelA = reference['ModelA']
            ModelB = reference['ModelB']

            if reference['category'] == 'Reason & Analysis':
                r1_rank_score = R1_Score_MAP['Reason_and_analysis']
            elif reference['category'] == 'Code & AI':
                r1_rank_score = R1_Score_MAP['Code_and_AI']
            else:
                r1_rank_score = R1_Score_MAP[reference['category']]

            choice = self.extract_winner(prediction, reference['lan'])
            detail = {
                'pred': prediction,
                'reference': reference,
                'correct': False
            }

            # calculate just when choice is not None
            if choice is not None:

                # calculate acc
                count += 1
                r1_gt = 'A' if reference['r1_gt'] == reference[
                    'ModelA'] else 'B'
                if r1_gt == choice:
                    correct += 1
                    detail['correct'] = True

                # calculate rank loss
                if choice == 'A':
                    if ModelA != 'gpt-4o-mini-2024-07-18':
                        if ModelA not in Model_dict:
                            Model_dict[ModelA] = 0
                        Model_dict[ModelA] += 1
                elif choice == 'B':
                    if ModelB != 'gpt-4o-mini-2024-07-18':
                        if ModelB not in Model_dict:
                            Model_dict[ModelB] = 0
                        Model_dict[ModelB] += 1

            details.append(detail)

        # calculate rank loss
        dict1 = dict(sorted(Model_dict.items()))
        dict2 = dict(sorted(r1_rank_score.items()))

        rank1 = self.get_rank_dict(dict1)
        rank2 = self.get_rank_dict(dict2)

        # 计算各维度差异
        rank_diffs = {m: abs(rank1[m] - rank2[m]) for m in rank1}
        score_diffs = {m: abs(dict1[m] - dict2[m]) for m in dict1}

        # 计算总差异（可自由调整权重）
        total_rank_diff = sum(rank_diffs.values())  # 例如原排名总差距 = 14
        total_score_diff = sum(score_diffs.values())  # 例如总分数差距 = 75
        alpha = 0.2  # 分数差异权重系数
        combined_diff = total_rank_diff + alpha * total_score_diff  # 例如综合差距 = 14 + 15 = 29

        # 计算归一化系数
        max_rank_diff = len(dict1) - 1  # 例如最大排名差 = 9
        max_score_diff = max(
            abs(d1 - d2)
            for d1, d2 in zip(dict1.values(), dict2.values()))  # 例如最大分数差 = 22

        # 计算归一化后的综合差距
        normalized_diffs = {
            m: abs(rank1[m] - rank2[m]) / max_rank_diff +
            abs(dict1[m] - dict2[m]) / max_score_diff
            for m in rank1
        }
        total_normalized_diff = sum(normalized_diffs.values()) / len(
            normalized_diffs.values()) * 100
        acc = 100 * correct / count
        final_score = (acc - total_normalized_diff + 100) / 2
        result = {
            'accuracy': acc,
            'rank_diff': total_rank_diff,
            'score_diff': total_score_diff,
            'normalized_diff': total_normalized_diff,
            'final_score': final_score,
            'details': details
        }
        return result
