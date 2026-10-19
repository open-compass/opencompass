# flake8: noqa

EXTRACT_PROMPT_CN = '''你是一个乐于助人的助手，任务是从给定的回答句子中提取精确的关键答案。你必须只提供提取的关键答案，不包括任何额外的文字。
—
我将为你提供一个问题、回答句子和问题类型。回答句子是对所提供问题的回应。利用提供的信息，你必须准确而精确地确定并从回答句子中提取预期的关键答案。请不要对问题发表主观看法。

对于单选题，答案应该是选项字母，例如 "A"；
对于多选题，答案应该是一个选项字母的列表，例如 ["A"] 或 ["A", "B", "C"]；
对于填空题，答案应该是一个填入空白处的答案列表，列表的数量应该与问题中的空白数量相同，同一空白的答案可能有多个，请在同一个 string 中用逗号隔开表示，如 ['sqrt(x) 且 x > 10', '1/2, 1/3', '1/4'] 代表问题包含三小问，第一小问包含取值范围信息，第二小问有两个答案，第三小问有一个答案。
对于解答题，类似填空题，答案应该是一个答案列表，每小问的答案间用逗号隔开，同样需要注意某些小问答案多个的情况。

如果回答句子提供了多个不同的答案，请仔细判断后面提供的答案是否是对前面答案的修正或修改。如果是这样，提取这个修正或修改后的答案作为最终答案。相反，如果回答句子在多个答案之间波动而没有明确的最终答案，你应该输出 [No valid answer]。
—
问题类型: {question_type}
原始问题: {question}
回答: {response}
提取的关键答案:
'''

EXTRACT_PROMPT_EN = '''You are a helpful assistant whose task is to extract precise key answers from given response sentences. You must only provide the extracted key answers without any additional text.
—
I will provide you with a question, a response sentence, and the question type. The response sentence is a reply to the provided question. Using the provided information, you must accurately and precisely identify and extract the expected key answers from the response sentence. Please do not provide subjective opinions about the question.

For single-choice questions, the answer should be the letter of the option, such as "A".
For multiple-choice questions, the answer should be a list of option letters, such as ["A"] or ["A", "B", "C"].
For fill-in-the-blank questions, the answer should be a list of answers to fill in the blanks. The number of items in the list should match the number of blanks in the question. If there are multiple answers for the same blank, separate them with a comma within the same string, like ['sqrt(x) and x > 10', '1/2, 1/3', '1/4'], which represents three sub-questions where the first sub-question includes a range, the second sub-question has two answers, and the third sub-question has one answer.
For problem-solving questions, similar to fill-in-the-blank questions, the answer should be a list of answers. Separate answers for different sub-questions with commas, and note that some sub-questions may have multiple answers.

If the response sentence provides multiple different answers, carefully determine whether a later provided answer is a correction or modification of an earlier answer. If so, extract this corrected or modified answer as the final answer. Conversely, if the response sentence fluctuates between multiple answers without a clear final answer, you should output [No valid answer].
—
Question type: {question_type}
Question: {question}
Output sentences: {response}
Key extracted answer:
'''

JUDGE_PROMPT_CN = '''请你作为一个数学阅卷专家，判断下面的答案是否与标准答案一致，即考生是否回答正确。下面是一些评判标准：
1. 有些答案可能包含多项内容，可能有单选题，多选题，填空题和问答题，只要答案与标准答案一致即可, 对于多选题和多个空的填空题，需要考生对应的选项或空都回答正确才算正确。
2. 有些答案可能通过不同的方式表达，比如有些答案可能是一个数学表达式，有些答案可能是一个文字描述，只要表达的意思一致即可。且有些公式通过不同的方式表达，但等价，也是正确的。
3. 你不需要重新计算问题答案，因为标准答案已经给出，只需要根据问题形式来判断考生的答案是否与标准答案一致，是否正确即可。

请你根据上述标准，判断下面的答案是否与标准答案一致，如果一致，请在最后输出\\boxed{{yes}}, 否则输出\\boxed{{no}}, 如果难以判断，请输出\\boxed{{no}}.
原问题：{question}
标准答案：{gold_answer}
考生答案：{answer}

分析：
'''

JUDGE_PROMPT_EN = '''Please act as an expert in grading mathematics exam papers, and judge whether the following answers match the standard answers, i.e., whether the examinee answered correctly. Here are some evaluation criteria:

1. Some answers may contain multiple parts, such as single-choice questions, multiple-choice questions, fill-in-the-blank questions, and problem-solving questions. As long as the answer matches the standard answer, it is considered correct. For multiple-choice questions and fill-in-the-blank questions with multiple blanks, the examinee must answer all corresponding options or blanks correctly to be considered correct.
2. Some answers may be expressed in different ways; for example, some answers may be mathematical expressions, while others may be textual descriptions. As long as the meaning conveyed is consistent, it is considered correct. Additionally, some formulas may be expressed differently but are equivalent, which is also considered correct.
3. You do not need to recalculate the problem answers, as the standard answers are already provided. You only need to judge whether the examinee's answer matches the standard answer based on the form of the question and whether it is correct.

Please judge whether the following answer matches the standard answer according to the above criteria. If they match, output \\boxed{{yes}}, otherwise output \\boxed{{no}}. If it is difficult to judge, also output \\boxed{{no}}.
Original Question: {question}
Standard Answer: {gold_answer}
Examinee's Answer: {answer}

Analysis:
'''

PROMPT_CN = '''下面是一个数学问题，请逐步推理，并把最终答案放置于\\boxed{{}}中。
{question}
'''

PROMPT_EN = '''Here is a math question, please reasoning step by step, and put your answer in \\boxed{{}}.
{question}
'''
