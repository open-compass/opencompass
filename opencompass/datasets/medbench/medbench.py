import json
import os.path as osp
import sys
from datasets import Dataset
from sklearn.metrics import classification_report
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .math_equivalence import is_equiv
from .post_process import parse_math_answer, parse_qa_multiple_answer

import evaluate
from nltk.translate.bleu_score import sentence_bleu
# # from bert_score import score
import re
from transformers import BasicTokenizer
from rouge_chinese import Rouge
basic_tokenizer = BasicTokenizer(tokenize_chinese_chars=True)

@LOAD_DATASET.register_module()
class MedBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, setting_name: str):
        path = get_data_path(path, local_mode=True)
        from .dataset_loader import load_dataset, load_dataset_as_result_schema

        assert setting_name in 'zero-shot', 'only support zero-shot setting'
        dataset_wo_label = load_dataset(name, setting_name, path)
        dataset_with_label = load_dataset_as_result_schema(name, path)
        dataset = []
        for d1, d2 in zip(dataset_wo_label, dataset_with_label):
            dataset.append({
                'id': d2.index,
                'problem_input': d1['context'],
                'label': d2.label,
            })
        dataset = Dataset.from_list(dataset)
        return dataset


@ICL_EVALUATORS.register_module()
class MedBenchEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        # predictions: [[]]
        # references: [[]]
        predictions = [parse_qa_multiple_answer(pred) for pred in predictions]
        details = []
        cnt = 0
        for pred, ref in zip(predictions, references):
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            if is_equiv(pred, ref):
                cnt += 1
                detail['correct'] = True
            details.append(detail)
        score = cnt / len(predictions) * 100
        return {'Accuracy': score, 'details': details}

def process_generated_results_CMeEE(pred_file):
    #   实体每类占一行，每行格式为 "[类型名称]实体：实体名称1，实体名称2，实体名称3\n"
    #                多个实体，用 ，符号分割
    structured_output = []
    answer_choices = ['药物', '设备', '医院科室', '微生物类', '身体部位', '医疗操作', '医学检验项目', '症状', '疾病']
    for pred in pred_file:
        list_entities = []
        for choice in answer_choices:
            for piece in re.split('\n', pred):
                if piece.startswith(f"{choice}"):
                    mentions = re.split(r"[,，]", piece.replace(f"{choice}:", "").replace(f"{choice}：", ""))
                    for ment in mentions:
                        list_entities.append({'type':choice, 'entity':ment})
        structured_output.append(list_entities)
    return structured_output

def process_generated_results_EMR(pred_file):
    structured_output = []
    regex = r"^(主诉|现病史|既往史|个人史|婚育史|家族史)[:：]([\s\S]+)$"
    for prediction in pred_file:
        entities: dict = {}
        if "\n\n" in prediction:
            blocks = prediction.split("\n\n")
        else:
            blocks = prediction.splitlines()
        for line in blocks:
            if match := re.match(regex, line.strip()):
                type_ = match[1]
                mention = match[2].strip()
                entities[type_] = mention
        structured_output.append(entities)
    return structured_output

def process_generated_results_CMeIE(pred_file):
    structured_output = []
    for line in pred_file:
        gen_output = line

        answer_choices = "相关（导致）、鉴别诊断、遗传因素、发病性别倾向、相关（症状）、手术治疗、预防、辅助检查、筛查、阶段、临床表现、风险评估因素、同义词、发病年龄、预后生存率、病史、传播途径、治疗后症状、药物治疗、辅助治疗、化疗、死亡率、放射治疗、病因、组织学检查、内窥镜检查、多发群体、并发症、实验室检查、就诊科室、病理生理、高危因素、发病率、多发地区、病理分型、影像学检查、转移部位、发病部位、相关（转化）、外侵部位、预后状况、发病机制、多发季节"
        re_choices = "|".join(re.escape(choice) for choice in answer_choices.split('、'))
        regex = (
        rf'关系[:：]["“]({re_choices})["”][,，]'
        r'头实体[:：]["“]([^"”]+)["”][,，]尾实体[:：]["“]([^"”]+)["”]'
        )

        list_spos = []
        list_answer_strs = gen_output.split("\n")
        for line in list_answer_strs:
            for item in re.finditer(regex, line):
                print(item)
            for match in re.finditer(regex, line):
                list_spos.append({"predicate": match[1], "subject": match[2], "object": match[3]})
                
        structured_output.append(list_spos)
    return structured_output

def process_generated_results_CDN(pred_file):
    structured_output = []
    answer_choices = json.load(open('./opencompass/datasets/medbench/entity_list.jsonl', 'r'))
    for line in pred_file:
        gen_output = line

        answer_str = gen_output.split("\n")[-1]
        answers = answer_str.split("，")
        answers = [w.strip() for w in answers if len(w.strip()) > 0]
        answers = [w for w in answers if w in answer_choices]
        answers = list(set(answers))
        answers = [
            {
                "entity": w,
                "type": "normalization",
            }
            for w in answers
        ]

        structured_output.append(answers)
    return structured_output

def process_generated_results_CDEE(pred_file):
    structured_output = []
    for prediction in pred_file:
        events: list[dict] = []
        for line in prediction.splitlines():
            if "主体词" in line:
                line = line.rstrip("。")
                kvs = line.split("；")
                kv_dict = dict(kv.split("：", maxsplit=1) for kv in kvs if "：" in kv)
                events.append({
                    "主体词": kv_dict.get("主体词", ""),
                    "发生状态": (
                        v
                        if (v := kv_dict.get("发生状态", "不确定")) in ("不确定", "否定")
                        else ""
                    ),
                    "描述词": (
                        v.split("，") if (v := kv_dict.get("描述词", "空")) != "空" else []
                    ),
                    "解剖部位": (
                        v.split("，")
                        if (v := kv_dict.get("解剖部位", "空")) != "空"
                        else []
                    ),
                })
        structured_output.append(events)
    return structured_output

def process_generated_results_CTC(pred_file):
    structured_output = []

    for line in pred_file:
        gen_output = line
        # 答案格式：直接回答分类标签
        answer_str = gen_output.strip()
        structured_output.append(answer_str)
    return structured_output

def process_generated_results_doc_parsing(pred_file):
    float_field_regex = r"(体温|脉搏|心率|收缩压|舒张压|呼吸)[^\d]*(\d+(?:\.\d+)?)"

    output = []
    for prediction in pred_file:
        entities = {
        "体温": "未扪及",
        "脉搏": "未扪及",
        "心率": "未扪及",
        "收缩压": "未扪及",
        "舒张压": "未扪及",
        "呼吸": "未扪及",
        "是否上腹部深压痛": None,
        "是否腹部反跳痛": None,
        "上腹部肿块": None,
    }
        for sentence in re.split("[，|。|\n]", prediction):
            for match in re.finditer(float_field_regex, prediction):
                entities[match[1]] = match[2]
            if "上腹部深压痛" in sentence:
                if re.search("是(?!否)|(?:^|[^不])存在|有", sentence):
                    entities["是否上腹部深压痛"] = "是"
                else:
                    entities["是否上腹部深压痛"] = "否"
            elif "腹部反跳痛" in sentence:
                if re.search("是(?!否)|(?:^|[^不])存在|有", sentence):
                    entities["是否腹部反跳痛"] = "是"
                else:
                    entities["是否腹部反跳痛"] = "否"
            elif "上腹部肿块" in sentence:
                if re.search("是(?!否)|(?:^|[^不])存在|有", sentence):
                    entities["上腹部肿块"] = "扪及"
                else:
                    entities["上腹部肿块"] = "未扪及"
        result = [
            {
                "type": "体温(℃)",
                "entity": entities["体温"],
            },
            {
                "type": "脉搏(次/分)",
                "entity": entities["脉搏"],
            },
            {
                "type": "心率(次/分)",
                "entity": entities["心率"],
            },
            {
                "type": "收缩压(mmHg)",
                "entity": entities["收缩压"],
            },
            {
                "type": "舒张压(mmHg)",
                "entity": entities["舒张压"],
            },
            {
                "type": "呼吸(次/分)",
                "entity": entities["呼吸"],
            },
        ]
        if entities["是否上腹部深压痛"]:
            result.append({
                "type": "是否上腹部深压痛",
                "entity": entities["是否上腹部深压痛"],
            })
        if entities["是否腹部反跳痛"]:
            result.append({
                "type": "是否腹部反跳痛",
                "entity": entities["是否腹部反跳痛"],
            })
        if entities["上腹部肿块"]:
            result.append({
                "type": "上腹部肿块",
                "entity": entities["上腹部肿块"],
            })

        output.append(result)
    return output

def process_generated_results_mrg(pred_file):
    structured_output = []
    regex = r"^(主诉|现病史|辅助检查|既往史|诊断|建议)[:：]([\s\S]+)$"
    for prediction in pred_file:
        entities = {}
        if "\n\n" in prediction:
            blocks = prediction.split("\n\n")
        else:
            blocks = prediction.splitlines()
        for line in blocks:
            if match := re.match(regex, line.strip()):
                type_ = match[1]
                mention = match[2].strip()
                entities[type_] = mention
        structured_output.append(entities)
    return structured_output

def calc_info_extract_task_scores(list_structured_predict, list_structured_golden):

    assert len(list_structured_golden) == len(list_structured_predict)

    tp = 0
    fp = 0
    fn = 0
    for samp_golden, samp_predict in zip(list_structured_golden, list_structured_predict):
        # samp_golden: [[{}]]
        answer_golden = samp_golden
        answer_predict = samp_predict
        # assert isinstance(answer_golden, list)
        # assert isinstance(answer_predict, list), "sample format is wrong!"

        set_golden = set()
        for inst in answer_golden:
            assert isinstance(inst, dict)
            keys = sorted(list(inst.keys()))
            inst = tuple([json.dumps(inst[w], ensure_ascii=False) for w in keys ])
            # inst = list(inst.items())
            # inst.sort()
            # inst = tuple(inst)

            set_golden.add(inst)

        set_predict = set()
        for inst in answer_predict:
            assert isinstance(inst, dict)
            keys = sorted(list(inst.keys()))

            inst = tuple([json.dumps(inst[w], ensure_ascii=False) for w in keys])

            set_predict.add(inst)

        tp += len(set_golden.intersection(set_predict))
        fp += len(set_predict.difference(set_golden))
        fn += len(set_golden.difference(set_predict))

    if tp:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    return precision, recall, f1

def calc_cls_task_scores(list_structured_golden,
                         list_structured_predict,
                         list_labels=None,
                         return_macro=False,
                         ):
    # types = list_labels
    # scores = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in list_labels + ["ALL"]}

    predictions = []
    ground_truths = []

    # Count GT relations and Predicted relations
    assert len(list_structured_golden) == len(list_structured_predict)
    n_sents = len(list_structured_golden)

    # Count TP, FP and FN per type
    for pred_samp, gt_samp in zip(list_structured_predict, list_structured_golden):

        pred_label = pred_samp
        gt_label = gt_samp
        # assert gt_label != ""
        if gt_label == "":
            get_label = list_labels[0]
        if pred_label == "":
            pred_label = list_labels[0]

        predictions.append(pred_label)
        ground_truths.append(gt_label)

    # metric
    cls_report = classification_report(
        ground_truths, predictions,
        output_dict=True,
        zero_division=0,
    )

    if return_macro:
        return cls_report["macro avg"]["precision"], \
               cls_report["macro avg"]["recall"], \
               cls_report["macro avg"]["f1-score"]
    else:
        return cls_report["weighted avg"]["precision"], \
               cls_report["weighted avg"]["recall"], \
               cls_report["weighted avg"]["f1-score"]

def calc_nlg_task_scores(list_structured_golden, list_structured_predict):

    assert len(list_structured_golden) == len(list_structured_predict)

    scores = []
    predictions = []
    references = []
    details = []
    for samp_golden, samp_predict in zip(list_structured_golden, list_structured_predict):

        answer_golden = samp_golden
        answer_predict = samp_predict

        if not (answer_predict and answer_golden):
            continue

        # basic tokenizer: 拆分中文字，保留英文单词
        answer_predict = basic_tokenizer.tokenize(answer_predict)
        answer_golden = basic_tokenizer.tokenize(answer_golden)
        answer_predict = " ".join(answer_predict).strip()
        answer_golden = " ".join(answer_golden).strip()
        if answer_golden.strip() == "":
            answer_golden = "无 。"
        if answer_predict.strip() == "":
            answer_predict = "无 。"

        predictions.append(answer_predict)
        references.append(answer_golden)

        details.append({'pred':answer_predict, 'answer':answer_golden, 'correct':False})

    rouge = Rouge()
    # bleu = evaluate.load('sacrebleu')
    scores = rouge.get_scores(predictions, references, avg=True)
    # scores_bleu = bleu.compute(predictions=predictions, references=references)

    rouge1 = scores["rouge-1"]["f"]
    rouge2 = scores["rouge-2"]["f"]
    rougeL = scores["rouge-l"]["f"]

    # bleu = sentence_bleu(references, predictions)

    # bert_score = []
    # for id in range(len(predictions)):
    #     P, R, F1 = score([predictions[i]], [references[i]], model_type='bert-base-chinese', lang="zh", verbose=True)
    #     bert_score.append(F1)
    # bert_score = float(sum(bert_score)) / float(len(bert_score))
    # return rougeL, bleu, bert_score
    return {'RougeL': rougeL, 'details':details}

def calc_scores_f1(dict_gt, dict_pred):
        details = []
        for gt, pred in zip(dict_gt, dict_pred):
            details.append({'pred':pred, 'answer':gt, 'correct':None})
        
        precision, recall, f1 = calc_info_extract_task_scores(dict_gt, dict_pred)
        return {'F1':f1, 'details':details}

def calc_scores_ctc(dict_gt, dict_pred):
    details = []
    for gt, pred in zip(dict_gt, dict_pred):
        details.append({'pred':pred, 'answer':gt, 'correct':None})

    gts = dict_gt
    preds = dict_pred
    
    precision, recall, f1 = calc_cls_task_scores(
        gts,
        preds,
        list_labels=['非上述类型', '疾病', '症状(患者感受)',
                    '体征(医生检测）', '怀孕相关', '肿瘤进展',
                    '疾病分期', '过敏耐受', '器官组织状态',
                    '预期寿命', '口腔相关', '药物',
                    '治疗或手术', '设备', '护理',
                    '诊断', '实验室检查', '风险评估',
                    '受体状态', '年龄', '特殊病人特征',
                    '读写能力', '性别', '教育情况',
                    '居住情况', '种族', '知情同意',
                    '参与其它试验', '研究者决定', '能力',
                    '伦理审查', '依存性', '成瘾行为',
                    '睡眠', '锻炼', '饮食', '酒精使用',
                    '性取向', '吸烟状况', '献血',
                    '病例来源', '残疾群体', '健康群体',
                    '数据可及性', "含有多个类别"],
        return_macro=True,
    )
    return {'Macro-F1':f1, 'details':details}
    
def calc_scores_nlg(dict_gt, dict_pred):
    
        # scores = {}
        scores = {'score':0, 'details':[]}
        success_flag = 1

        gts = dict_gt
        preds = dict_pred
        # if not len(gts) == len(preds):
        #     success_flag = 0
        # try:
        return calc_nlg_task_scores(gts, preds)    

@ICL_EVALUATORS.register_module()
class MedBenchEvaluator_CMeEE(BaseEvaluator):

    def score(self, predictions, references):
        predictions = process_generated_results_CMeEE(predictions)
        return calc_scores_f1(predictions, references)

@ICL_EVALUATORS.register_module()
class MedBenchEvaluator_DBMHG(BaseEvaluator):

    def score(self, predictions, references):
        predictions = process_generated_results_EMR(predictions)
        return calc_scores_f1(predictions, references)

@ICL_EVALUATORS.register_module()
class MedBenchEvaluator_IMCS_V2_MRG(BaseEvaluator):

    def score(self, predictions, references):
        # predictions = process_generated_results_mrg(predictions)
        references_revise = []
        for item in references:
            temp_ref = ''
            for sub_item in item:
                temp_ref += sub_item['type'] + '：' + sub_item['entity'] + '\n'
            references_revise.append(temp_ref)
        return calc_nlg_task_scores(references_revise, predictions)

@ICL_EVALUATORS.register_module()
class MedBenchEvaluator_CMeIE(BaseEvaluator):

    def score(self, predictions, references):
        predictions = process_generated_results_CMeIE(predictions)
        return calc_scores_f1(predictions, references)

@ICL_EVALUATORS.register_module()
class MedBenchEvaluator_CHIP_CDEE(BaseEvaluator):

    def score(self, predictions, references):
        predictions = process_generated_results_CDEE(predictions)
        return calc_scores_f1(predictions, references)

@ICL_EVALUATORS.register_module()
class MedBenchEvaluator_CHIP_CDN(BaseEvaluator):

    def score(self, predictions, references):
        predictions = process_generated_results_CDN(predictions)
        return calc_scores_f1(predictions, references)

@ICL_EVALUATORS.register_module()
class MedBenchEvaluator_CHIP_CTC(BaseEvaluator):

    def score(self, predictions, references):
        predictions = process_generated_results_CTC(predictions)
        return calc_scores_ctc(predictions, references)

@ICL_EVALUATORS.register_module()
class MedBenchEvaluator_Doc_parsing(BaseEvaluator):

    def score(self, predictions, references):
        # predictions = process_generated_results_doc_parsing(predictions)
        references_revise = []
        for item in references:
            temp_ref = ''
            for sub_item in item:
                temp_ref += sub_item['type'] + '：' + sub_item['entity'] + '\n'
            references_revise.append(temp_ref)
        return calc_nlg_task_scores(references_revise, predictions)

@ICL_EVALUATORS.register_module()
class MedBenchEvaluator_NLG(BaseEvaluator):

    def score(self, predictions, references):
        # predictions = process_generated_results_med(predictions)
        return calc_scores_nlg(predictions, references)

@ICL_EVALUATORS.register_module()
class MedBenchEvaluator_Cloze(BaseEvaluator):

    def score(self, predictions, references):
        # predictions: [[]]
        # references: [[]]
        # predictions = [parse_qa_multiple_answer(pred) for pred in predictions]
        details = []
        cnt = 0

        for pred, ref in zip(predictions, references):
            detail = {'pred':pred, 'answer':ref, 'correct':False}
            
            if sum([item in pred for item in ref]) == len(ref):
                cnt += 1
                detail['correct'] = True
            details.append(detail)
        score = cnt / len(predictions) * 100
        return {'Accuracy': score, 'details': details}

@ICL_EVALUATORS.register_module()
class MedBenchEvaluator_TF(BaseEvaluator):

    def score(self, predictions, references):
        # predictions: [[]]
        # references: [[]]
        # predictions = [parse_qa_multiple_answer(pred) for pred in predictions]
        details = []
        cnt = 0

        for pred, ref in zip(predictions, references):
            
            if '不' in pred or '否' in pred:
                cur_pred = '不可以'
            else:
                cur_pred = '可以'

            detail = {'pred':cur_pred, 'answer':ref, 'correct':False}

            if cur_pred == ref:
                cnt += 1
                detail['correct'] = True
            
            details.append(detail)

        score = cnt / len(predictions) * 100
        return {'Accuracy': score, 'details': details}