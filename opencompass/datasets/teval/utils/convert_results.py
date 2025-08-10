import mmengine
import os
import argparse
import numpy as np
# np.set_printoptions(precision=1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str)
    args = parser.parse_args()
    return args

def convert_results(result_path):
    result = mmengine.load(result_path)
    instruct_list = [(result['instruct_json']['json_format_metric'] + result['instruct_json']['json_args_em_metric']) / 2, \
                     (result['instruct_json']['string_format_metric'] + result['instruct_json']['string_args_em_metric']) / 2]
    plan_list = [result['plan_str']['f1_score'], result['plan_json']['f1_score']]
    reason_list = [result['reason_str']['thought'], result['rru_json']['thought']]
    retrieve_list = [result['retrieve_str']['name'], result['rru_json']['name']]
    understand_list = [result['understand_str']['args'], result['rru_json']['args']]
    review_list = [result['review_str']['review_quality'], result['review_str']['review_quality']]

    final_score = [np.mean(instruct_list), np.mean(plan_list), np.mean(reason_list), \
                   np.mean(retrieve_list), np.mean(understand_list), np.mean(review_list)]
    overall = np.mean(final_score)
    final_score.insert(0, overall)
    name_list = ['Overall', 'Instruct', 'Plan', 'Reason', 'Retrieve', 'Understand', 'Review']
    print("Cut Paste Results: ", np.array(final_score) * 100)
    for i in range(len(name_list)):
        print("%s: %.1f" % (name_list[i], final_score[i]*100), end='\t')


if __name__ == '__main__':
    args = parse_args()
    convert_results(args.result_path)
