from datasets import Dataset, load_dataset
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from ..base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.datasets.prompt import *
import os
import json
import os.path as osp
import random
from typing import Dict, List, Optional
import numpy as np
import mmengine
import arrow
import os, glob, random, json, re, time
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from mmengine.config import ConfigDict
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import ICL_PROMPT_TEMPLATES
from opencompass.utils import build_dataset_from_cfg, build_model_from_cfg
from opencompass.utils.logging import get_logger
import re

csimpleqa_judge_prompt_new = """
请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】或【未尝试】。

首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。
以下是【正确】的答复示例：
```
问题：贝拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：Malia Obama and Sasha Obama
模型预测2：玛丽亚和萨沙
模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认
模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。玛丽亚出生于1998年7月4日，萨莎出生于2001年6月10日。
```
这些答复均为【正确】，因为：
    - 完整地包含了标准答案中的重要信息。
    - 不包含任何与标准答案矛盾的信息。
    - 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。
    - 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。

以下是【错误】的答复示例：
```
问题：巴拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：玛丽亚
模型预测2：玛丽亚、萨莎和苏珊
模型预测3：巴拉克·奥巴马没有孩子
模型预测4：我认为是玛丽亚和萨莎。或者是玛丽亚和杰基。或者是乔伊和玛丽亚。
模型预测5：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有三个孩子。
模型预测6：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？
```
这些答复均为【错误】，因为：
    - 答复中包含与标准答案矛盾的事实陈述。即使在陈述中略带保留（例如：“可能是”，“虽然我不确定，但我认为”），也视为错误。

以下是【未尝试】的答复示例：
```
问题：巴拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：我不知道。
模型预测2：我需要更多关于您所指奥巴马的上下文。
模型预测3：不查阅网络我无法回答这个问题，不过我知道巴拉克·奥巴马有两个孩子。
模型预测4：巴拉克·奥巴马有两个孩子。我知道其中一个叫玛丽亚，但我不确定另一个的名字。
```
这些答复均为【未尝试】，因为：
    - 没有包含标准答案中的重要信息。
    - 回复中没有与标准答案矛盾的陈述。

另外注意以下几点：
- 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题“金山铁路黄浦江特大桥的全长是多少米？”，标准答案为“3518.17”：
    - 预测答案“3518”、“3518.1”、“3518.17”均为【正确】。
    - 预测答案“3520”和“3600”均为【错误】。 
    - 预测答案“大约3500米”和“超过3000米”被视为【未尝试】，因为它们既不确认也不与标准答案矛盾。
- 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。
    - 例如，考虑问题“菱镁矿的主要化学成分是什么？”标准答案为“碳酸镁（MgCO3）”。“碳酸镁”或“MgCO3”均视为【正确】答案。
- 如果从问题中明显可以推断出预测答案省略的信息，那么算作正确。
    - 例如，问题“巴鲁米尼的努拉吉遗迹在1997年被联合国教科文组织列为世界文化遗产，那么这遗址在哪个地区？”标准答案为“意大利撒丁岛”，预测答案“撒丁岛”被视为【正确】。
- 如果能明显看出名字翻译版本不同但是是同一个人也认为正确。
    - 例如，如果标准答案是“Robinson”，那么回答鲁滨逊或者鲁滨孙均正确。

下面是一个新的问题示例。请只回复A、B、C之一，不要道歉或纠正自己的错误，只需要评估该回答。
```
问题: {question}
正确答案: {target}
预测答案: {predicted_answer}
```

将此新问题的预测答案评定为以下之一：
A:【正确】
B:【错误】
C:【未尝试】

只返回字母"A"、"B"或"C"，无须添加其他文本。
""".strip()


@TEXT_POSTPROCESSORS.register_module('qa_base_preprocess')
def qa_base_preprocess(text: str) -> str:
    text = text.split("问题：")[0].strip()
    return text

@LOAD_DATASET.register_module()
class CsimpleqaDataset(BaseDataset):

    @staticmethod
    def check(path, part = "test"):
        cur_path = os.path.join(path, f"{part}.jsonl")
        lines = open(cur_path, "r", encoding='utf-8').readlines()
        writer = open(cur_path, "w", encoding='utf-8')
        for line in lines:
            data = json.loads(line)
            for k, v in data.items():
                if not isinstance(v, str) and k != "messages":
                    data[k] = str(v)
            writer.write(json.dumps(data, ensure_ascii=False) + '\n')

    @staticmethod
    def load(**kwargs):
        # AElogDataset.check(kwargs['path'], "test")
        dataset = load_dataset(**kwargs)
        split = 'test'
        raw_data = []
        for i in range(len(dataset[split])):
            question = dataset[split]['question'][i]
            cur_system_prompt = "你是一个智能助手。"
            messages = [{"role": "system", "content": cur_system_prompt}, {"role": "user", "content": question}]
            judge_system_prompt = "你是一个智能助手，请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。"
            csimpleqa_judge_prompt_f = csimpleqa_judge_prompt_new.format(question = question, target = dataset[split]['answer'][i], predicted_answer = "{prediction}")
            raw_data.append({
                'question': question,
                'gold_ans': dataset[split]['answer'][i],
                'messages': messages,
                'system_prompt': judge_system_prompt,
                'prompt_template': csimpleqa_judge_prompt_f
            })
        dataset[split] = Dataset.from_list(raw_data)
        dataset["train"] = Dataset.from_list(raw_data)
        return dataset


class CsimpleqaEvaluator(BaseEvaluator):

    def __init__(
        self,
        prompt_template: ConfigDict,
        judge_cfg: ConfigDict,
        output_path: str,
        meta_review_prompt_template: Optional[ConfigDict] = None,
        pack_all_predictions: Optional[bool] = False,
        dataset_cfg: Optional[ConfigDict] = None,
        pred_postprocessor: Optional[ConfigDict] = None,
    ) -> None:
        self.output_path = output_path
        out_dir, out_name = osp.split(output_path)
        if not out_dir:
            out_dir = './'

        self.prompt_tmpl = ICL_PROMPT_TEMPLATES.build(prompt_template)
        if meta_review_prompt_template is not None:
            self.meta_review_prompt_tmpl = ICL_PROMPT_TEMPLATES.build(
                meta_review_prompt_template)

        max_out_len = judge_cfg.get('max_out_len', None)
        batch_size = judge_cfg.get('batch_size', None)
        model = build_model_from_cfg(model_cfg=judge_cfg)
        self.inferencer = GenInferencer(model,
                                        max_out_len=max_out_len,
                                        batch_size=batch_size,
                                        output_json_filepath=out_dir,
                                        output_json_filename=out_name)
        self.logger = get_logger()
        self.dataset_cfg = dataset_cfg
        self.pack_all_predictions = pack_all_predictions

    def qa_base_preprocess(self, text: str) -> str:
        text = text.split("问题：")[0].strip()
        return text
    

    def score(self,
              predictions,
              judgements: Optional[List] = None,
              references: Optional[List] = None,
              meta: Optional[bool] = False,
              infer_order: Optional[str] = 'random') -> Dict:

        dup_indices = []
        if isinstance(predictions, list):
            """Apply to multi-model comparison."""
            references = [{} for _ in range(len(predictions[0]['model_preds']))
                          ] if references is None else references
            predictions, references = order_preds_and_record_references(
                predictions, references, infer_order)

            # calculate dupicated predictions numbers
            total_predictions_num = len(predictions[0])

            # since there is impossible that two models response same pattern in multi-round chat, so we just check dup for single chat
            if isinstance(predictions[0][0], str):
                for i in range(len(predictions[0])):
                    check = [sub[i] for sub in predictions]
                    if len(set(check)) == 1:
                        dup_indices.append(i)

        elif isinstance(predictions, dict):
            """Apply to single-model scoring."""
            references = [{} for _ in range(len(predictions['model_preds']))
                          ] if references is None else references
            predictions = [predictions['model_preds']]
        # Due to the rarity of identical predictions, we have temporarily disabled the plagiarism detection feature.
        dup_indices = []
        predictions = [[self.qa_base_preprocess(pre) for pre in predictions[0]]]
        if len(dup_indices) != 0:
            # remove dupicated predictions
            for index in sorted(dup_indices, reverse=True):
                for sublist in predictions:
                    del sublist[index]
                del references[index]

        pred_dict = {}
        if isinstance(
                predictions[0][0], str
        ):  #single chat for format like [['xxx', 'xxxx'], ['xxx', 'xxxx']]
            for i in range(len(predictions)):
                key = 'prediction' if i == 0 else f'prediction{i + 1}'
                gold_key = 'obj_gold'
                pred_dict[key] = predictions[i]
                pred_dict[gold_key] = references
            if judgements:
                for i in range(len(judgements)):
                    key = 'judgement' if i == 0 else f'judgement{i + 1}'
                    pred_dict[key] = judgements[i]['model_preds']
                    for j in range(len(references)):
                        references[j]['judge_model' +
                                      str(i + 1)] = judgements[i]['model_name']

        elif isinstance(
                predictions[0][0], list
        ):  
            if self.pack_all_predictions:
                for i in range(len(predictions)):
                    key = 'prediction' if i == 0 else f'prediction{i + 1}'
                    pred_dict[key] = predictions[i]
            else:
                for i in range(len(predictions)):
                    multiround_predictions = extract_dicts(predictions[i])
                    for j in range(len(multiround_predictions)):
                        key = 'prediction' if i == 0 else f'prediction{i}'
                        key += '_r' + str(j + 1)
                        pred_dict[key] = multiround_predictions[j]
            if judgements:
                raise NotImplementedError(
                    'Not applied meta-reivew judge on multi-round dataset')
        else:
            raise NotImplementedError(
                f'{predictions[0][0]} with type {type(predictions[0][0])}, please check the postprocess you add to the prediction string is right or not, we suggest to return an empty string but not None'
            )
        print("self.dataset_cfg: ", self.dataset_cfg)
        if self.dataset_cfg:
            print("self.dataset_cfg: ", self.dataset_cfg)
            dataset = build_dataset_from_cfg(self.dataset_cfg)

            if infer_order == 'double':
                new_ds = {
                    k: dataset.test[k] * 2
                    for k in dataset.test.column_names
                }
                dataset.reader.dataset['test'] = Dataset.from_dict(new_ds)

            if len(dup_indices) != 0:
                remaining_indices = [
                    idx for idx in range(len(dataset.test))
                    if idx not in dup_indices
                ]
                dataset.reader.dataset['test'] = dataset.test.select(
                    remaining_indices)
                print(
                    f'Among total {total_predictions_num} predictions, there are {len(dup_indices)} predictions totally same, which are removed!'
                )
            for k, v in pred_dict.items():
                dataset.reader.dataset['test'] = dataset.test.add_column(k, v)
                dataset.reader.input_columns.append(k)

            if references:
                dataset.reader.input_columns.append('reference')
                dataset.reader.dataset['test'] = dataset.test.add_column(
                    'reference', references)
            print("dataset.reader.input_columns:", dataset.reader.input_columns)
        else:
            # build a default dataset just for comparison
            from opencompass.datasets.lmeval import LMEvalDataset
            input_columns = list(pred_dict.keys())
            print("input_columns: ", input_columns)
            if references:
                input_columns.append('reference')
            dataset = LMEvalDataset(reader_cfg=dict(
                input_columns=input_columns,
                output_column=None,
                train_split='test'),
                                    reference=references,
                                    **pred_dict)
        print("dataset.reader.input_columns :", dataset.reader.input_columns)
        dataset.reader.output_column = 'reference'
        retriever = ZeroRetriever(dataset)
        print("self.prompt_tmpl: ", self.prompt_tmpl)
        print("self.meta: ", meta)
        print("dataset.reader.input_columns: ", dataset.reader.input_columns)
        if meta:
            self.inferencer.inference(
                retriever=retriever,
                prompt_template=self.meta_review_prompt_tmpl)
        else:
            self.inferencer.inference(retriever=retriever,
                                      prompt_template=self.prompt_tmpl, input_json_filepath=self.dataset_cfg.path, stage = 'eval')
        self.output_path = "/".join(self.output_path.split("/")[:-3]) + "/judge.jsonl"
        return self.postprocess(self.output_path)

    def calculate_accuracies(self, group):
        total_questions = len(group)
        total_correct = group[group['score'] == "A"].shape[0]
        total_incorrect = group[group['score'] == "B"].shape[0]
        total_not_attempted = group[group['score'] == "C"].shape[0]
        
        total_correct_accuracy = total_correct / total_questions if total_questions > 0 else 0
        total_incorrect_accuracy = total_incorrect / total_questions if total_questions > 0 else 0
        total_not_attempted_accuracy = total_not_attempted / total_questions if total_questions > 0 else 0
        
        total_given_attempted_accuracy = total_correct / (total_correct + total_incorrect) if (total_correct + total_incorrect) > 0 else 0
        
        f1 = 2 * total_given_attempted_accuracy * total_correct_accuracy / (total_given_attempted_accuracy+ total_correct_accuracy) if (total_given_attempted_accuracy+ total_correct_accuracy) > 0 else 0

        return pd.Series({'correct': total_correct_accuracy, 'incorrect': total_incorrect_accuracy, 'not_attempted': total_not_attempted_accuracy, "given_attempted_accuracy": total_given_attempted_accuracy, "F1": f1})

    def postprocess(self, output_path: str) -> Dict:
        datas = []
        lines = open(output_path, "r", encoding='utf-8').readlines()
        for line in lines:
            data = json.loads(line)
            judge = data.get("judge", "")
            score = "C"
            try:
                match = re.search(r"(A|B|C)", judge)
                score = match.group(0) if match else "C" 
            except:
                score = "C"
            data['score'] = score
            datas.append(data)
        df = pd.json_normalize(datas)
        accuracy_df = self.calculate_accuracies(df)
        results = {
            'primary_category': ['Overall'],
            'correct': [accuracy_df['correct']],
            'incorrect': [accuracy_df['incorrect']],
            'not_attempted': [accuracy_df['not_attempted']],
            'given_attempted_accuracy': [accuracy_df['given_attempted_accuracy']],
            'F1': [accuracy_df['F1']],
        }
        overall_row = pd.DataFrame(results)
        accuracy_df = df.groupby('primary_category').apply(self.calculate_accuracies).reset_index()
        final_df = pd.concat([overall_row, accuracy_df], ignore_index=True)

        final_df.to_csv(output_path.replace(".jsonl", ".csv"), index=False)
     
        return results


