import random

def randomize_preds_and_record_references(predictions, references, seed=1):
    # 为了保证结果的可重复性，我们设置一个固定的随机种子
    random.seed(seed)
    list_of_preds = [[] for _ in range(len(predictions))]
    for i in range(len(predictions[0]['model_preds'])):
        # 对于每一个问题，我们都获取所有模型的预测
        preds = [[pred['model_preds'][i], pred['model_name']] for pred in predictions]

        # 然后对这些预测进行随机排序
        random.shuffle(preds)
        #print(preds)
        # 将排序后的预测添加到最终列表中
        for j in range(len(preds)):
            #pred = preds[j]
            list_of_preds[j].append(preds[j][0])
            references[i][f'answer{j+1}'] = preds[j][1]
    return list_of_preds, references

predictions = [{'model_name': 'baichuan2-7b-chat-hf', 'model_preds': ['球最初会向上（垂直）方向行进。', '根据描述，这些书的顺序如下：1. 蓝皮书（最右边）2. 红皮书（从左数第二本）3. 黄皮书（左数第三本）4. 绿皮书（左边）', '1. 剪指甲 2. 穿袜子 3. 穿鞋 4. 系鞋带']}, {'model_name': 'qwen-7b-chat-hf', 'model_preds': ['球最初会向上行进。', '绿红黄橙蓝', '正确的顺序应该是：剪指甲、穿袜子、穿鞋、系鞋带。']},
{'model_name': 'qwen-14b-chat-hf', 'model_preds': ['上行进。', '绿红黄橙蓝', '剪指甲、穿袜子、穿鞋、系鞋带。']}]

references = [{},{},{},{}]

list_of_preds, references = randomize_preds_and_record_references(predictions, references)

print(list_of_preds)
print(references)