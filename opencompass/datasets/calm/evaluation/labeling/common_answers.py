import re

common_true_list = [
    'answer (yes or no?): yes', 'answer (yes or no? ): yes',
    'answer (yes or no ?): yes', 'answer (yes or no ? ): yes', 'answer is yes',
    "\"yes\"", 'say yes', 'as follows:\n\nyes', 'answer: yes',
    'answer is: yes', 'answer is:\n\nyes', 'answer is:\nyes',
    "answer is \"yes\"", 'should be yes', 'is yes', 'chose yes', '\n\nyes',
    'the correct answer is yes', 'is identified', 'is identifiable',
    'does cause', '答案是:是', '答案是:\n\n是', '“是”', '答案：是', '答案是:\n是', '答案:是',
    '答案是是', "\"是\"", '是的', '方法1比方法2更准确', '答：\n\n是', '答案为：是', '```\nyes\n\n```',
    'answer (yes or no?):yes', 'answer (yes or no? ):yes',
    'answer (yes or no ?):yes', 'answer (yes or no ? ):yes', 'output: yes',
    'answer (yes or no?): yes', 'answer is yes', "\"yes\"", 'say yes',
    'as follows:\n\nyes', 'answer: yes', 'answer is: yes', 'answer is:\n\nyes',
    'answer is:\nyes', '“是”', '答案：是', 'answer (yes or no?): yes',
    'answer is yes', "\"yes\"", 'answer: yes', 'answer is: yes',
    'answer is:\n\nyes', 'answer is:\nyes', '答案是:是', '答案是:\n\n是', '答案是:\n是',
    '答案:是', '答案是是', "\"是\"", '是的', '答案: 是', '回答是:是',
    'answer (yes or no?): yes', 'answer is yes', "\"yes\"", 'answer: yes',
    'answer is: yes', 'answer is:\n\nyes', 'answer is:\nyes', '答案是:是',
    '答案是:\n\n是', '答案是:\n是', '答案:是', '答案是是', "\"是\"", '是的', '答案: 是',
    'answer (yes or no?): yes', 'answer (yes or no ?): yes', 'answer is yes',
    "\"yes\"", 'answer: yes', 'answer is: yes', 'answer is:\n\nyes', 'so yes',
    'therefore, yes', 'answer is:\nyes', 'method 1 is correct',
    'correct to use method 1', 'chose yes', 'yes, it would', '答案是:是',
    '答案是:\n\n是', '答案是:\n是', '答案:是', '答案是是', "\"是\"", '是的', '方法1比方法2更准确',
    '答案: 是', '回答是:是', '答案为“是', '答案是“是', '答案应该是“是', '答案都是“是', '答案(是)', '答案是肯定',
    '答案是：是', '答案：是', '答案是:“是', '答案为:是', '答案是「是', '答案为 “是', '存在因果关系', '答案是真',
    '因此是“是', '答案为“有', '答案是是', '答案为“(是', "答案为\"\"是", '答案是:“是', "答案应为:\"是",
    "答案应为\"是", '答案为:是', "答案是:\"是", "答案应该是\"是", '答案为“yes', '具有因果关系', '答案是 “是',
    '答案“是', '答案必须是“yes', '答案处为“是', '答案应是“是', '答案為“是', '答案可以是“是', '答案的是“是',
    '答案为「是', "案为“\"是", "答案为 \"是", '答案是有', '答案是： 是', '答案为：是', '答案是对', '答案是：是',
    '答案是：\n是', '答案应为“是', '答案：是', '答案应该是“是', '答案(是或否？)：是的', "答案\"是\"",
    "答案都是\"是", '答案为是', '答案为 “是', "答案为\"是", '答案为“是', "答案是\"是\"", '答案是“是',
    '答案是“是”', 'answer (yes )', 'answering your query, yes', 'so yes',
    'henceforth answering yes', 'hence yes', 'answer (yes)', 'in short, yes',
    'hence - yes', 'correct response should be yes', 'thus, yes',
    'in short - yes', '答案是：\n\nyes', 'leading us to believe yes!',
    'hence answering yes', 'therefore should read - yes', 'hence y es',
    'therefore, yes', 'therefore yes',
    'the correct response should be marked as “yes.', 'thus - yes',
    "the answer is \"yes.\"", "therefore, the answer is \"yes,\"",
    'answer (yes or no ? ) : yes', 'answe: yes', "thus answering 'yes'",
    'thus answering yes', 'thereby answering yes', 'answer would thus be yes',
    "so answering 'yes'", "hence answering 'yes'", "therefore answering 'yes",
    'confirming our answer yes', 'an answer for this question would be yes',
    'answer would be: yes', 'implying a yes answer', 'making the answer yes',
    'incident does have a causal relationship',
    'the cause and effect relationship exists',
    'there is a direct cause relationship', 'must have a causal relationship',
    'answer would be yes', 'a causal relationship exists between',
    'answer(yes', 'answer for this question is yes', 'answer (yes',
    'answer here is `yes`', 'answer might be yes', 'answer is a yes',
    'the answer yes', 'henceforth – yes', 'thus indicating yes',
    'hence indicating yes', "it's safe to say yes", "hence it's 'yes'",
    "thus answering 'yes’", 'so it’s yes', 'thus it can be said yes',
    'the correct response is yes', 'answering the question with a yes',
    "the correct answer would be \"yes", "the answer is \"yes”",
    "answer \"yes", 'the answer as yes', 'the answer to the question yes',
    'the answer is causality', 'the answer is yes', "the answer is \"yes",
    '答案是:是'
]
common_false_list = [
    'answer (yes or no?): no', 'answer (yes or no? ): no',
    'answer (yes or no ?): no', 'answer (yes or no ? ): no', 'answer is no',
    "\"no\"", 'say no', 'as follows:\n\nno', 'answer: no', 'answer is: no',
    'answer is:\n\nno', 'answer is:\nno', 'should be no', "answer is \"no\"",
    'is no', 'chose no', '\n\nno', 'the correct answer is no.',
    'is not identified', '答案是:否', '答案是：否', '答：否', '答案是:\n\n否', '“否”', '答案：否',
    '答案是:\n否', '答案:否', '答案是否', "\"否\"", '答案：不是', '回答是:否', '答：\n\n否', '不会导致',
    '```\nno\n\n```', 'answer (yes or no?):no', 'answer (yes or no? ):no',
    'answer (yes or no ?):no', 'answer (yes or no ? ):no', 'output: no',
    'answer (yes or no?): no', 'answer is no', "\"no\"", 'say no',
    'as follows:\n\nno', 'answer: no', 'answer is: no', 'answer is:\n\nno',
    'answer is:\nno', '“否”', '答案：否', '答案：不是', 'answer (yes or no?): no',
    'answer is no', "\"no\"", 'answer: no', 'answer is: no',
    'answer is:\n\nno', 'answer is:\nno', 'does not cause', '答案是:否',
    '答案是:\n\n否', '答案是:\n否', '答案:否', '答案是否', "\"否\"", '并没有导致', '回答是:否', '答案: 否',
    '不会导致', '回答是:\n\n否', '回答是:\n否', 'answer (yes or no?): no', 'answer is no',
    "\"no\"", 'answer: no', 'answer is: no', 'answer is:\n\nno',
    'answer is:\nno', '答案是:否', '答案是:\n\n否', '答案是:\n否', '答案:否', '答案是否', "\"否\"",
    '回答是:否', '答案: 否', 'answer (yes or no?): no', 'answer is no', "\"no\"",
    'answer: no', 'answer is: no', 'answer is:\n\nno', 'answer is:\nno',
    'method 2 is correct', 'correct to use method 2', 'chose no', '答案是:否',
    '答案是:\n\n否', '答案是:\n否', '答案:否', '答案是否', "\"否\"", '回答是:否', '方法2比方法1更准确',
    '方法2', '答案是:否', '答案是:\n\n否', '答案是:\n否', '答案:否', '答案是否', "\"否\"", '并没有导致',
    '回答是:否', '答案: 否', '不会导致', '回答是:\n\n否', '回答是:\n否', '答案为“否', '答案是“否',
    '答案是：\n\n- 否', '答案是：否', '答案为：否', '答案：否', '答案(是或否？)：否', '答案是不', '答案应该是“否',
    '答案应为否', "答案应该是\"否", '答案应为“否', '答案为否', 'answering your query - no',
    "the answer is \"no.\"", 'the answer is therefore no',
    'hence answering no', "hence answering 'no'", 'answer should read : no',
    'therefore answer would be no', "answers should read 'no",
    'answer would need to be no', 'answering your above query : no',
    'answer would be no', "therefore, answering 'no", 'answer:no',
    'answer should remain no', 'the answer to this question would be no',
    'answer is:no', "answer is therefore \"no.\"", 'making the answer no',
    'answer(no)', 'answer is, no', "answer might be \"no.\"",
    'answer it as no', 'should be the answer no', 'answering no',
    "thus answering 'no'", 'thus, no', "therefore 'no'",
    'the answer can be no', 'answer is “no', 'the answer is mostly no',
    'answer is probably not', "answer is \"no", '答案是“否”', '答案（是或否？）：否',
    "答案是\"否\"", "答案为\"否\"", '答案是否', '答案为“否', '答案为“不', '答案为“没有', '答案“否',
    '答案为“非”', '答案为“无”', '答案为”否', "答案为 \"否", '答案为否', '答案是\\”否', '答案应该是“否',
    '答案是：\nno', '答案是：\n否', '答案是：\n不', '答案：否', '答案应为“非', "答案\"否", '答案为**否',
    '答案在“否', '答案可能为“否', '答案返回“否', "答案为\"否", '答案是“不', '答案应该为“否', "答案为'否",
    '答案为不存  在', '答案应为“否', '答案为《否', '答案是“无', '答案为\\“否', '答案将是“否', '答案还是“否',
    '答案：“不', '答案 为“否', '答案应该是否', 'the answer is no', '不存在“因果”关系', "答案应为\"否",
    "答案应该是\"否", '答案是:否', '答案为:否', "答案选择\"否", "答案是:\"否", "答案应该为\"否", "答案应为\"否",
    '答案选择为:否', '答案为 “否', '答案为“非', '答案为“没'
]
common_option_1_list = [
    'answer (option 1 or 2 or 3 or 4 ?): option 1',
    '(option 1 or 2 or 3 or 4?): option 1',
    'answer (option 1 or 2 or 3 ?): option 1',
    '(option 1 or 2 or 3?): option 1',
    'answer (option 1 or option 2 ?): option 1',
    'answer (option 1 or option 2?): option 1', 'should be 1', 'is 1',
    'option 1', 'answer is 1', 'option one',
    "the correct answer is \"option 1", '正确答案是选项一', '答案为选项一', '应该选择选项一',
    '答案：选项一', '答案: 选项一', '答案:选项一', '答案是选项一', '选项一是正确', '选择选项一', '我认为选项一',
    '我的回答是选项一', '我的回答是:选项一', 'option 1', 'answer is 1', 'option one', '答案：选项一',
    'option  1', 'option#1', '选项1是最符合', '答案是选项1', '答案为选项1', '选项1是正确',
    '答案应该是选项1', '答案是选项 1', 'answer is therefore option 1', '答案为选项一', '答案（选项一）',
    '选项一正确', '选选项一', '答案选项一', '即选项一', '答案是：\n选项一', '答案为选项一', '是选项一', '选项一是正确',
    '选项一为正确', '选项一的答案是正确', '答案为选项一', '答案:选项一', '是:选项一', '答案: 选项一'
]
common_option_2_list = [
    'answer (option 1 or option 2 ?): option 2',
    'answer (option 1 or option 2?): option 2',
    'answer (option 1 or 2 or 3 or 4 ?): option 2',
    '(option 1 or 2 or 3 or 4?): option 2',
    'answer (option 1 or 2 or 3 ?): option 2',
    '(option 1 or 2 or 3?): option 2', 'should be 2', 'is 2', 'option 2',
    'answer is 2', 'option two', "the correct answer is \"option 2",
    '正确答案是选项二', '答案为选项二', '应该选择选项二', '答案：选项二', '答案: 选项二', '答案:选项二', '答案是选项二',
    '选项二是正确', '选择选项二', '我认为选项二', '我的回答是选项二', '我的回答是:选项二', 'option 2',
    'answer is 2', 'option two', '答案：选项二', 'option ##two##', '选项2是满足',
    '答案是选项2', '答案是选项 2', '答案为选项二', '答案是选项二', '是选项二', '答案（选项二）', '选项二正确',
    '选选项二', '答案选项二', '即选项二', '答案是：\n选项二', '答案为选项二', '选项二是正确', '选项二为正确',
    '答案为选项二', '答案:选项二', '是:选项二', '答案: 选项二'
]
common_option_3_list = [
    'answer (option 1 or 2 or 3 or 4 ?): option 3',
    '(option 1 or 2 or 3 or 4?): option 3',
    'answer (option 1 or 2 or 3 ?): option 3',
    '(option 1 or 2 or 3?): option 3', 'should be 3', 'is 3', 'option 3',
    'answer is 3', 'option three', '正确答案是选项三', '答案为选项三', '应该选择选项三', '答案：选项三',
    '答案: 选项三', '答案:选项三', '答案是选项三', '选项三是正确', '选择选项三', '我认为选项三', '我的回答是选项三',
    '我的回答是:选项三', 'option 3', 'answer is 3', 'option three', '答案：选项三', '答案是选项3',
    '选项 3 是正确', '选项3是正确', '答案是选项 3', '答案为选项三', '答案是选项三', '是选项三', '选项三正确',
    '选选项三', '答案选项三', '即选项三', '答案是：\n选项三', '答案为选项三', '选项三是正确', '选项三为正确',
    '答案为选项三', '答案:选项三', '是:选项三', '答案: 选项三'
]
common_option_4_list = [
    'answer (option 1 or 2 or 3 or 4 ?): option 4',
    '(option 1 or 2 or 3 or 4?): option 4', 'should be 4', 'is 4', 'option 4',
    'answer is 4', 'option four', '正确答案是选项四', '答案为选项四', '应该选择选项四', '答案：选项四',
    '答案: 选项四', '答案:选项四', '答案是选项四', '选项四是正确', '选择选项四', '我认为选项四', '我的回答是选项四',
    '我的回答是:选项四', 'option 4', 'answer is 4', 'option four', '答案：选项四', '答案是选项4',
    '选项 4 是正确', '选项4是正确', '答案是选项 4', '答案为选项四', '答案是选项四', '是选项四', '选项四正确',
    '选选项四', '答案选项四', '即选项四', '答案是：\n选项四', '答案为选项四', '选项四是正确', '选项四为正确',
    '答案为选项四', '答案:选项四', '是:选项四', '答案: 选项四'
]

common_start_true_dict = {
    1: ['答案（是或否？）：是', '答案（是或否？）：- 是', '答案（是或否？）：\n\n是', '有'],
    3: [
        'answer (yes or no?): yes', 'answer (yes or no? ): yes',
        'answer (yes or no ?): yes', 'answer (yes or no ? ): yes',
        'answer (yes or no?): \n\nyes', 'answer (yes or no? ): \n\nyes',
        'answer (yes or no ?): \n\nyes', 'answer (yes or no ? ): \n\nyes',
        'answer (yes or no?):yes', 'answer (yes or no? ):yes',
        'answer (yes or no ?):yes', 'answer (yes or no ? ):yes',
        'answer (yes or no?): - yes', 'answer (yes or no? ): - yes',
        'answer (yes or no ?): - yes', 'answer (yes or no ? ): - yes', '答案为“是”'
    ],
    4: [
        'answer (yes or no?): true', 'answer (yes or no? ): true',
        'answer(yes;', 'answer(yes)'
    ],
    5: ['answer (yes )']
}
common_start_false_dict = {
    1: [
        '答案（是或否？）：否', '答案（是或否？）：- 否', '答案（是或否？）：\n\n否', '答案（是或否？）：不',
        '答案（是或否？）：- 不', '无'
    ],
    2: [
        '答案（是或否？）：不是',
        '答案（是或否？）：- 不是',
        'answer (yes or no?): no',
        'answer (yes or no? ): no',
        'answer (yes or no ?): no',
        'answer (yes or no ? ): no',
        'answer (yes or no?):no',
        'answer (yes or no? ):no',
        'answer (yes or no ?):no',
        'answer (yes or no ? ):no',
        'answer (yes or no?): \n\nno',
        'answer (yes or no? ): \n\nno',
        'answer (yes or no ?): \n\nno',
        'answer (yes or no ? ): \n\nno',
        'answer (yes or no?): - no',
        'answer (yes or no? ): - no',
        'answer (yes or no ?): - no',
        'answer (yes or no ? ): - no',
    ],
    3: ['答案为“否”', 'answer (no)', 'answer(no)'],
    4: ['answer (no )', 'answe r(no )'],
    5: [
        'answer (yes or no?): false',
        'answer (yes or no? ): false',
    ],
}

common_start_op1_dict = {
    1: [
        '答案（选项一或选项二或选项三？）：选项一', '答案（选项一或选项二或选项三？）： 选项一', '答案（选项一或选项二或选项三？）：一',
        '答案（选项一或选项二或选项三？）： 一', 'answer (option 1 or 2 or 3?) : option 1',
        'answer (option 1 or 2 or 3?) : option1',
        'answer (option 1 or 2 or 3?):1', 'answer (option 1 or 2 or 3?): 1',
        '答案（选项一或选项二或选项三或选项四？）：选项一', '答案（选项一或选项二或选项三或选项四？）： 选项一',
        '答案（选项一或选项二或选项三或选项四？）：一', '答案（选项一或选项二或选项三或选项四？）： 一',
        'answer (option 1 or 2 or 3 or 4?) : option 1',
        'answer (option 1 or 2 or 3 or 4?) : option1',
        'answer (option 1 or 2 or 3 or 4?):1',
        'answer (option 1 or 2 or 3 or 4?): 1', '答案（选项一或选项二？）：选项一',
        'answer (option 1 or option 2) : option 1',
        'answer (option 1 or option 2) : option1', 'answer: option 1',
        'the correct answer is option 1', 'the answer is option 1'
    ],
    3: [
        'answer (option 1 or 2 or 3?) : option one',
        'answer (option 1 or 2 or 3 or 4?) : option one',
        'answer (option 1 or option 2) : option one'
    ],
}
common_start_op2_dict = {
    1: [
        '答案（选项一或选项二或选项三？）：选项二', '答案（选项一或选项二或选项三？）： 选项二', '答案（选项一或选项二或选项三？）：二',
        '答案（选项一或选项二或选项三？）： 二', 'answer (option 1 or 2 or 3?) : option 2',
        'answer (option 1 or 2 or 3?) : option2',
        'answer (option 1 or 2 or 3?):2', 'answer (option 1 or 2 or 3?): 2',
        '答案（选项一或选项二或选项三或选项四？）：选项二', '答案（选项一或选项二或选项三或选项四？）： 选项二',
        '答案（选项一或选项二或选项三或选项四？）：二', '答案（选项一或选项二或选项三或选项四？）： 二',
        'answer (option 1 or 2 or 3 or 4?) : option 2',
        'answer (option 1 or 2 or 3 or 4?) : option2',
        'answer (option 1 or 2 or 3 or 4?):2',
        'answer (option 1 or 2 or 3 or 4?): 2', '答案（选项一或选项二？）：选项二',
        'answer (option 1 or option 2) : option 2',
        'answer (option 1 or option 2) : option2', 'answer: option 2',
        'the correct answer is option 2', 'the answer is option 2'
    ],
    3: [
        'answer (option 1 or 2 or 3?) : option two',
        'answer (option 1 or 2 or 3 or 4?) : option two',
        'answer (option 1 or option 2) : option two'
    ],
}
common_start_op3_dict = {
    1: [
        '答案（选项一或选项二或选项三？）：选项三',
        '答案（选项一或选项二或选项三？）： 选项三',
        '答案（选项一或选项二或选项三？）：三',
        '答案（选项一或选项二或选项三？）： 三'
        'answer (option 1 or 2 or 3?) : option 3',
        'answer (option 1 or 2 or 3?) : option3',
        'answer (option 1 or 2 or 3?):3',
        'answer (option 1 or 2 or 3?): 3',
        '答案（选项一或选项二或选项三或选项四？）：选项三',
        '答案（选项一或选项二或选项三或选项四？）： 选项三',
        '答案（选项一或选项二或选项三或选项四？）：三',
        '答案（选项一或选项二或选项三或选项四？）： 三'
        'answer (option 1 or 2 or 3 or 4?) : option 3',
        'answer (option 1 or 2 or 3 or 4?) : option3',
        'answer (option 1 or 2 or 3 or 4?):3',
        'answer (option 1 or 2 or 3 or 4?): 3',
    ],
    5: [
        'answer (option 1 or 2 or 3?) : option three',
        'answer (option 1 or 2 or 3 or 4?) : option three'
    ],
}
common_start_op4_dict = {
    1: [
        '答案（选项一或选项二或选项三或选项四？）：选项四',
        '答案（选项一或选项二或选项三或选项四？）： 选项四',
        '答案（选项一或选项二或选项三或选项四？）：四',
        '答案（选项一或选项二或选项三或选项四？）： 四'
        'answer (option 1 or 2 or 3 or 4?) : option 4',
        'answer (option 1 or 2 or 3 or 4?) : option4',
        'answer (option 1 or 2 or 3 or 4?):4',
        'answer (option 1 or 2 or 3 or 4?): 4',
    ],
    4: ['answer (option 1 or 2 or 3 or 4?) : option four']
}


# some shared answer processing functions in mathematically related tasks
def is_numeric(value):
    try:
        float(value)
        return True
    except Exception:
        return False


def add_quotes_to_unquoted(json_str):
    # This regex looks for words that are not surrounded by quotes.
    return re.sub(r'(?<=[:,])\s*([\w_]+)\s*(?=[,:\]})])', r' "\1" ', json_str)


def change_quotation(json_str):
    json_str = re.sub(r'“', '"', json_str)
    json_str = re.sub(r'”', '"', json_str)
    json_str = re.sub(r'\'', '"', json_str)
    return json_str
