import re
import os
import subprocess
import tempfile

"""
Task: legal document grammar correction
Metric: F0.5 score
文书校对
"""
def compute_wsjd(data_dict):
    origins, references, predictions = [], [], []
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        if isinstance(question, list):
            question = question[0]['prompt']
        start = question.index('句子：\n') + 4
        origins.append(re.sub(r'\n|\t', '', question[start:].split('\n')[0]))
        # truncate predictions >5 tokens longer than the reference
        prediction = re.sub(r'\n|\t', '', prediction)
        if len(prediction) - len(answer) > 5:
            prediction = prediction[:len(answer) + 5]
        if len(prediction) == 0:
            prediction = "无内容"
        predictions.append(prediction)
        references.append(re.sub(r'\n|\t', '', answer))

    #generate input files for ChERRANT
    preds = [f'{i} \t {origin} \t {prediction} \n' for i, (origin, prediction) in enumerate(zip(origins, predictions))]
    golds = [f'{i} \t {origin} \t {reference} \n' for i, (origin, reference) in enumerate(zip(origins, references))]

    now_path = os.path.abspath(os.getcwd())
    utils_path = os.path.abspath(os.path.join(__file__, '..', '..', 'utils'))
    os.chdir(utils_path)
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp_pred_file, \
            tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp_gold_file:
        tmp_pred_file.writelines(preds)
        tmp_gold_file.writelines(golds)

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.system(f'python3 parallel_to_m2.py -f {tmp_pred_file.name} -o {tmp_pred_file.name}.m2 -g char')
    os.system(f'python3 parallel_to_m2.py -f {tmp_gold_file.name} -o {tmp_gold_file.name}.m2 -g char')
    output = subprocess.check_output(
        f"python3 compare_m2_for_evaluation.py -hyp {tmp_pred_file.name}.m2 -ref {tmp_gold_file.name}.m2", shell=True)
    score = float(output.decode().split('\t')[-1].split('\n')[0])
    #remove prediction files
    os.remove(tmp_pred_file.name)
    os.remove(tmp_gold_file.name)
    os.remove(f"{tmp_pred_file.name}.m2")
    os.remove(f"{tmp_gold_file.name}.m2")
    os.chdir(now_path)
    return {"score": score}
