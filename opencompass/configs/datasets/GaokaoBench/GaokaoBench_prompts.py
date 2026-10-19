
MCQ_prompts = [
    {
        'type': 'single_choice',
        'keyword': '2010-2022_Math_II_MCQs',
        'prefix_prompt': '请你做一道数学选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：',
        'comment': '',
    },
    {
        'type': 'single_choice',
        'keyword': '2010-2022_Math_I_MCQs',
        'prefix_prompt': '请你做一道数学选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：',
        'comment': '',
    },
    {
        'type': 'single_choice',
        'keyword': '2010-2022_History_MCQs',
        'prefix_prompt': '请你做一道历史选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：',
    },
    {
        'type': 'single_choice',
        'keyword': '2010-2022_Biology_MCQs',
        'prefix_prompt': '请你做一道生物选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：',
    },
    {
        'type': 'single_choice',
        'keyword': '2010-2022_Political_Science_MCQs',
        'prefix_prompt': '请你做一道政治选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：',
    },
    {
        'type': 'multi_choice',
        'keyword': '2010-2022_Physics_MCQs',
        'prefix_prompt': '请你做一道物理选择题。\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出所有符合题意的答案，并写在【答案】和<eoa>之间。\n例如：【答案】 AB <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】... <eoa>\n请你严格按照上述格式作答。\n',
    },
    {
        'type': 'single_choice',
        'keyword': '2010-2022_Chemistry_MCQs',
        'prefix_prompt': '请你做一道化学选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：',
    },
    {
        'type': 'single_choice',
        'keyword': '2010-2013_English_MCQs',
        'prefix_prompt': '请你做一道英语选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：',
    },
    {
        'type': 'multi_question_choice',
        'keyword': '2010-2022_Chinese_Modern_Lit',
        'prefix_prompt': '请你做一道语文阅读理解题，其中包含三个小题。\n请你一步一步思考。每一题你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：（1）【答案】 A <eoa>\n（2）【答案】 B <eoa>\n请你严格按照上述格式作答。\n',
    },
    {
        'type': 'multi_question_choice',
        'keyword': '2010-2022_English_Fill_in_Blanks',
        'prefix_prompt': '请你做一道英语完形填空题,其中包含二十个小题。\n请你一步一步思考。每一题你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：（1）【答案】 A <eoa>\n（2）【答案】 B <eoa>\n请你严格按照上述格式作答。\n',
    },
    {
        'type': 'five_out_of_seven',
        'keyword': '2012-2022_English_Cloze_Test',
        'prefix_prompt': '请回答下面的问题，将符合题意的五个选项的字母写在【答案】和<eoa>之间，例如“【答案】 A B C D E <eoa>\n请严格按照上述格式作答。\n',
    },
    {
        'type': 'multi_question_choice',
        'keyword': '2010-2022_Geography_MCQs',
        'prefix_prompt': '请你做一道地理选择题，其中包含两到三个小题。\n请你一步一步思考。每一题你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：（1）【答案】 A <eoa>\n（2）【答案】 B <eoa>\n请你严格按照上述格式作答。\n',
    },
    {
        'type': 'multi_question_choice',
        'keyword': '2010-2022_English_Reading_Comp',
        'prefix_prompt': '请你做一道英语阅读理解题，其中包含三到五个小题。\n请你一步一步思考。每一题你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：（1）【答案】 A <eoa>\n（2）【答案】 B <eoa>\n请你严格按照上述格式作答。\n',
    },
    {
        'type': 'multi_question_choice',
        'keyword': '2010-2022_Chinese_Lang_and_Usage_MCQs',
        'prefix_prompt': '请你做一道语文选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n（1）【解析】 ... <eoe>\n【答案】 ... <eoa>\n（2）【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。如果不止一道题，请分别作答\n题目如下：',
    },
]
FBQ_prompts = [
    {
        'type': 'cloze',
        'keyword': '2010-2022_Math_I_Fill-in-the-Blank',
        'prefix_prompt': '请解答下面的数学填空题\n仔细阅读题目，解答其中的问题，请你一步步思考并将思考过程写在【解析】和<eoe>之间。请把你的答案写在【答案】和<eoa>之间。\n完整的题目回答格式如下：\n【解析】 ...<eoe>\n【答案】...<eoa>\n请你严格按照上述格式作答。\n题目如下:',
        'comment': '',
    },
    {
        'type': 'cloze',
        'keyword': '2010-2022_Math_II_Fill-in-the-Blank',
        'prefix_prompt': '请解答下面的数学填空题\n仔细阅读题目，解答其中的问题，请你一步步思考并将思考过程写在【解析】和<eoe>之间。请把你的答案写在【答案】和<eoa>之间。\n完整的题目回答格式如下：\n【解析】 ...<eoe>\n【答案】...<eoa>\n请你严格按照上述格式作答。\n题目如下:',
        'comment': '',
    },
    {
        'type': 'cloze',
        'keyword': '2010-2022_Chinese_Language_Famous_Passages_and_Sentences_Dictation',
        'prefix_prompt': '请回答下面的语文填空题\n请你仔细阅读题目，先找到题目对应的中国名篇，再从名篇中找到合适的句子填写到题目的空白处。请你将思考过程写在【解析】和<eoe>之间，将最终答案写在【答案】和<eoa>之间。\n完整的题目回答格式如下：\n（1）【解析】 ...<eoe>\n【答案】...<eoa>\n（2）【解析】 ...<eoe>\n【答案】...<eoa>\n请严格按照上述格式作答，如果不止一道题，请分别作答。\n题目如下:',
        'comment': '',
    },
    {
        'type': 'cloze',
        'keyword': '2014-2022_English_Language_Cloze_Passage',
        'prefix_prompt': '请回答下面的英语短文填词题\n仔细阅读题目，空白处请填入一个适当单词或者括号内单词的正确形式。请你一步步思考，将思考过程写在【解析】和<eoe>之间，将最终答案写在【答案】和<eoa>之间。\n完整的题目回答格式如下：\n（1）【解析】 ...<eoe>\n【答案】...<eoa>\n（2）【解析】 ...<eoe>\n【答案】...<eoa>\n请严格按照上述格式作答，如果不止一道题，请分别作答。\n题目如下:',
        'comment': '',
    },
]
OEQ_prompts = [
    {
        'type': 'subjective',
        'keyword': '2010-2022_Geography_Open-ended_Questions',
        'prefix_prompt': '请解答下面的地理解答题\n仔细阅读题目并充分结合你已有的知识，解答其中的问题，请你一步步思考并将思考过程写在【解析】和<eoe>之间。你的答案请写在【答案】和<eoa>之间\n完整的题目回答格式如下：\n(1)【解析】 ...<eoe>\n【答案】...<eoa>\n (2)【解析】 ...<eoe>\n【答案】...<eoa>\n请你严格按照上述格式作答，如果不止一道题，请分别作答。\n题目如下：',
        'comment': '',
    },
    {
        'type': 'subjective',
        'keyword': '2010-2022_Chemistry_Open-ended_Questions',
        'prefix_prompt': '请解答下面的化学解答题\n仔细阅读题目并充分结合你已有的知识，解答其中的问题，请你一步步思考并将思考过程写在【解析】和<eoe>之间。请把你的答案写在【答案】和<eoa>之间\n完整的题目回答格式如下：\n(1)【解析】 ...<eoe>\n【答案】...<eoa>\n (2)【解析】 ...<eoe>\n【答案】...<eoa>\n请你严格按照上述格式作答，如果不止一道题，请分别作答。\n题目如下:',
        'comment': '',
    },
    {
        'type': 'subjective',
        'keyword': '2010-2022_Math_I_Open-ended_Questions',
        'prefix_prompt': '请解答下面的数学解答题\n仔细阅读题目并充分结合你已有的知识，解答其中的问题，请你一步步思考并将思考过程写在【解析】和<eoe>之间。请把你的答案写在【答案】和<eoa>之间，答案需要有完整的解题步骤。\n完整的题目回答格式如下：\n(1)【解析】 ...<eoe>\n【答案】...<eoa>\n (2)【解析】 ...<eoe>\n【答案】...<eoa>\n请你严格按照上述格式作答，如果不止一道题，请分别作答。\n题目如下:',
        'comment': '',
    },
    {
        'type': 'subjective',
        'keyword': '2010-2022_History_Open-ended_Questions',
        'prefix_prompt': '请解答下面的历史解答题\n仔细阅读材料和题目，并充分结合你已有的知识，解答其中的问题。请你一步步思考并将思考过程写在【解析】和<eoe>之间。请把你的答案写在【答案】和<eoa>之间\n完整的题目回答格式如下：\n(1)【解析】 ...<eoe>\n【答案】...<eoa>\n (2)【解析】 ...<eoe>\n【答案】...<eoa>\n请你严格按照上述格式作答，如果不止一道题，请分别作答。\n题目如下:',
        'comment': '',
    },
    {
        'type': 'subjective',
        'keyword': '2010-2022_Biology_Open-ended_Questions',
        'prefix_prompt': '请解答下面的生物解答题\n仔细阅读题目并充分结合你已有的知识，解答其中的问题，请你一步步思考并将思考过程写在【解析】和<eoe>之间。请把你的答案写在【答案】和<eoa>之间,同一小题的答案用\t分隔开。\n完整的题目回答格式如下：\n(1)【解析】 ...<eoe>\n【答案】...\t...<eoa>\n (2)【解析】 ...<eoe>\n【答案】...\t...<eoa>\n请你严格按照上述格式作答，如果不止一道题，请分别作答。\n题目如下:',
        'comment': '',
    },
    {
        'type': 'subjective',
        'keyword': '2010-2022_Math_II_Open-ended_Questions',
        'prefix_prompt': '请解答下面的数学解答题\n仔细阅读题目并充分结合你已有的知识，解答其中的问题，请你一步步思考并将思考过程写在【解析】和<eoe>之间。请把你的答案写在【答案】和<eoa>之间，答案需要有完整的解题步骤。\n完整的题目回答格式如下：\n(1)【解析】 ...<eoe>\n【答案】...<eoa>\n (2)【解析】 ...<eoe>\n【答案】...<eoa>\n请你严格按照上述格式作答，如果不止一道题，请分别作答。\n题目如下:',
        'comment': '',
    },
    {
        'type': 'subjective',
        'keyword': '2010-2022_Physics_Open-ended_Questions',
        'prefix_prompt': '请解答下面的物理解答题，仔细阅读题目，注意其中可能含有单选题和多选题。请你一步步思考并将思考过程写在【解析】和<eoe>之间。请把你的最终答案写在【答案】和<eoa>之间。选择题你要从选项中选出符合题意的答案，例如“【答案】A <eoa>”。\n完整的题目回答格式如下：（1）【解析】 ...<eoe>\n【答案】 ...<eoa>\n (2)【解析】 ...<eoe>\n【答案】...<eoa>\n请你严格按照上述格式作答。如果不止一道题，请分别作答。\n题目如下:',
        'comment': '',
    },
    {
        'type': 'subjective',
        'keyword': '2010-2022_Political_Science_Open-ended_Questions',
        'prefix_prompt': '请解答下面的政治解答题\n仔细阅读材料和题目，并充分结合你已有的知识，解答其中的问题，请你一步步思考并将思考过程写在【解析】和<eoe>之间。请把你的答案写在【答案】和<eoa>之间\n完整的题目回答格式如下：\n(1)【解析】 ...<eoe>\n【答案】...<eoa>\n (2)【解析】 ...<eoe>\n【答案】...<eoa>\n请你严格按照上述格式作答，如果不止一道题，请分别作答。\n题目如下:',
        'comment': '',
    },
    {
        'type': 'correction',
        'keyword': '2012-2022_English_Language_Error_Correction',
        'prefix_prompt': '请解答下面的英语短文改错题，仔细阅读题目并充分结合你你已有的知识，找出其中10处需要改动的地方。请你一步步思考，把修改后的短文写在【答案】和<eoa>之间。\n完整的题目回答格式如下：【答案】 ...<eoa>\n 请你严格按照上述格式作答。\n题目如下:',
        # "prefix_prompt": [
        #     "请解答下面的英语短文改错题，仔细阅读题目并充分结合你你已有的知识，找出其中10处需要改动的地方。请你一步步思考，把修改后的短文写在【答案】和<eoa>之间。\n完整的题目回答格式如下：【答案】 ...<eoa>\n 请你严格按照上述格式作答。\n题目如下:",
        #     "请比较下面两篇短文，找到第二篇和第一篇的10处不同，每处不同只涉及一个单词，请将结果写在【答案】和<eoa>之间。例如：【答案】1. 将play改为plays\n 2.增加了the\n ... <eoa>\n 完整的题目回答格式如下：【答案】(1) ... \n (2) ...\n ...(10) ...\n<eoa>\n请你严格按照上述格式作答。\n短文如下:"
        # ],
        'comment': '',
    },
    {
        'type': 'subjective',
        'keyword': '2010-2022_Chinese_Language_Ancient_Poetry_Reading',
        'prefix_prompt': '请解答下面的语文古代诗歌阅读题，仔细阅读题目，注意其中可能含有单选题和多选题。请你一步步思考并将最终答案写在【答案】和<eoa>之间。选择题你要从选项中选出符合题意的答案，例如“【答案】A <eoa>”。\n完整的题目回答格式如下：（1）【答案】 ...<eoa>\n (2)【答案】...<eoa>\n请你严格按照上述格式作答，如果不止一道题，请分别作答。\n题目如下:',
        'comment': '',
    },
    {
        'type': 'subjective',
        'keyword': '2010-2022_Chinese_Language_Practical_Text_Reading',
        'prefix_prompt': '请解答下面的语文实用类文本阅读，仔细阅读题目，注意其中可能含有单选题和多选题。请你一步步思考并将最终答案写在【答案】和<eoa>之间。选择题你要从选项中选出符合题意的答案，例如“【答案】A <eoa>”。\n完整的题目回答格式如下：（1）[答案】 ...<eoa>\n (2)【答案】...<eoa>\n请你严格按照上述格式作答，如果不止一道题，请分别作答。\n题目如下:',
        'comment': '',
    },
    {
        'type': 'subjective',
        'keyword': '2010-2022_Chinese_Language_Literary_Text_Reading',
        'prefix_prompt': '请解答下面的语文文学类文本阅读，仔细阅读题目，注意其中可能含有单选题和多选题。请你一步步思考并将最终答案写在【答案】和<eoa>之间。选择题你要从选项中选出符合题意的答案，例如“【答案】A <eoa>”。\n完整的题目回答格式如下：（1）[答案】 ...<eoa>\n (2)【答案】...<eoa>\n请你严格按照上述格式作答,如果不止一道题，请分别作答。\n题目如下:',
        'comment': '',
    },
    {
        'type': 'subjective',
        'keyword': '2010-2022_Chinese_Language_Classical_Chinese_Reading',
        'prefix_prompt': '请解答下面的语文文言文阅读，仔细阅读题目，前三题是单选题，最后一题要将文言文翻译为现代汉语。请你一步步思考并把最终答案写在【答案】和<eoa>之间。选择题你要从选项中选出符合题意的答案，例如“【答案】A <eoa>”。翻译题把翻译后的现代汉语句子写在【答案】后面，例如”【答案】今天天气很好 <eoa>”\n完整的题目回答格式如下：（1）[答案】 ...<eoa>\n (2)【答案】...<eoa>\n请你严格按照上述格式作答，如果不止一道题，请分别作答。\n题目如下:',
        'comment': '',
    },
    {
        'type': 'subjective',
        'keyword': '2010-2022_Chinese_Language_Language_and_Writing_Skills_Open-ended_Questions',
        'prefix_prompt': '请解答下面的语文解答题，仔细阅读题目，注意其中可能含有选择题。请你一步步思考并将思考过程写在【解析】和<eoe>之间。请把你的最终答案写在【答案】和<eoa>之间。选择题你要从选项中选出符合题意的答案，例如“【答案】A <eoa>”。\n完整的题目回答格式如下：（1）【解析】 ...<eoe>\n【答案】 ...<eoa>\n (2)【解析】 ...<eoe>\n【答案】...<eoa>\n请你严格按照上述格式作答。如果不止一道题，请分别作答。\n题目如下:',
        'comment': '',
    },
]
