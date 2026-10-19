from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import CircularEvaluator, AccEvaluator
from opencompass.datasets import MathBenchDataset, mathbench_postprocess
from opencompass.utils.text_postprocessors import first_option_postprocess


single_choice_prompts = {
    'single_choice_cn_with_reasoning': '以下是一道关于数学的单项选择题，请你一步一步推理，并在最后用“所以答案为选项X”给出答案，其中“X”为选项A，B，C，D中你认为正确的选项。下面是你要回答的问题\n{question}\n让我们一步一步思考：\n',
    'single_choice_cn': '以下是一道关于数学的单项选择题，请你直接回答正确答案的选项序号。\n下面是你要回答的题目：\n{question}\n答案选项：',
    'single_choice_en_with_reasoning': "Here is a multiple-choice question about mathematics. Please reason through it step by step, and at the end, provide your answer option with 'Therefore, the correct answer is option X', Where 'X' is the correct option you think from A，B，C，D. Here is the question you need to answer:\n{question}\nLet's think step by step:",
    'single_choice_en': 'Here is a multiple-choice question about mathematics. Please provide the correct answer option directly.\nHere is the question you need to answer:\n{question}\nAnswer option:',
}

cloze_prompts = {
    'cloze_cn': [
                dict(role='HUMAN', prompt='Q: 林中有15棵树。林务工人员今天将在林中种植树木。完成后，将有21棵树。林务工人员今天种植了多少棵树？'),
                dict(role='BOT', prompt='A: 我们从15棵树开始。后来有21棵树。差值必定是他们种植的树木数量。所以，他们必须种植了21 - 15 = 6棵树。答案是 6\n'),
                dict(role='HUMAN', prompt='Q: 如果停车场有3辆车，又有2辆车进来，停车场里有多少辆车？'),
                dict(role='BOT', prompt='A: 停车场已经有3辆车。又进来了2辆车。现在有3 + 2 = 5辆车。答案是 5\n'),
                dict(role='HUMAN', prompt='Q: 黎恩有32块巧克力，她的妹妹有42块。如果他们吃了35块，他们总共剩下多少块？'),
                dict(role='BOT', prompt='A: 黎恩有32块巧克力，Leah的妹妹有42块。这意味着原本有32 + 42 = 74块巧克力。被吃掉了35块。所以他们总共还剩下74 - 35 = 39块巧克力。答案是 39\n'),
                dict(role='HUMAN', prompt='Q: 杰森有20个棒棒糖。他给丹妮一些棒棒糖。现在Jason只剩下12个棒棒糖。杰森给丹妮多少个棒棒糖？'),
                dict(role='BOT', prompt='A: 杰森有20个棒棒糖。因为他现在只剩下12个，所以他必须把剩下的都给了丹妮。他给丹妮的棒棒糖数量必定是20 - 12 = 8个。答案是 8\n'),
                dict(role='HUMAN', prompt='Q: 莎莎有五个玩具。在圣诞节，他从他的爸爸和妈妈那里各得到了两个玩具。现在他有多少个玩具？'),
                dict(role='BOT', prompt='A: 她有5个玩具。他从妈妈那里得到了2个，所以之后他有5 + 2 = 7个玩具。然后他从爸爸那里得到了2个，所以总共他有7 + 2 = 9个玩具。答案是 9\n'),
                dict(role='HUMAN', prompt='Q: 服务器房间里有九台电脑。从周一到周四每天增加五台电脑。现在服务器房里有多少台电脑？'),
                dict(role='BOT', prompt='A: 从周一到周四有4天。每天增加5台电脑。这意味着总共增加了4 * 5 = 20台电脑。一开始有9台电脑，所以现在有9 + 20 = 29台电脑。答案是 29\n'),
                dict(role='HUMAN', prompt='Q: 迈克尔有58个高尔夫球。星期二，他丢失了23个高尔夫球。星期三，他又丢失了2个。星期三结束时他还剩下多少个高尔夫球？'),
                dict(role='BOT', prompt='A: 迈克尔一开始有58个球。星期二他丢失了23个，所以之后他还剩下58 - 23 = 35个球。星期三他又丢失了2个，所以现在他还剩下35 - 2 = 33个球。答案是 33\n'),
                dict(role='HUMAN', prompt='Q: 奥利弗有23美元。她用每个3美元的价格买了五个百吉饼。她还剩下多少钱？'),
                dict(role='BOT', prompt='A: 她以每个3美元的价格买了5个百吉饼。这意味着她在百吉饼上花费了5 * 3 = 15美元。她一开始有23美元，所以现在她还剩下23 - 15 = 8美元。答案是 8\n'),
                dict(role='HUMAN', prompt='Q: {question}'),
                dict(role='BOT', prompt='A: {answer}'),
                ],
    'cloze_en': [
                dict(role='HUMAN', prompt='Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?'),
                dict(role='BOT', prompt='A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.\n'),
                dict(role='HUMAN', prompt='Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?'),
                dict(role='BOT', prompt='A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.\n'),
                dict(role='HUMAN', prompt='Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?'),
                dict(role='BOT', prompt="A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.\n"),
                dict(role='HUMAN', prompt='Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?'),
                dict(role='BOT', prompt='A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.\n'),
                dict(role='HUMAN', prompt='Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?'),
                dict(role='BOT', prompt='A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.\n'),
                dict(role='HUMAN', prompt='Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?'),
                dict(role='BOT', prompt='A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.\n'),
                dict(role='HUMAN', prompt='Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?'),
                dict(role='BOT', prompt='A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.\n'),
                dict(role='HUMAN', prompt='Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?'),
                dict(role='BOT', prompt='A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.\n'),
                dict(role='HUMAN', prompt='Q: {question}'),
                dict(role='BOT', prompt='A: {answer}\n'),
]}
cloze_prompts['refine_cloze_cn'] = cloze_prompts['cloze_cn']

mathbench_sets = {
    'college': ['single_choice_cn', 'cloze_en'],
    'high': ['single_choice_cn', 'single_choice_en'],
    'middle': ['single_choice_cn'],
    'primary': ['cloze_cn'],
    'primary_refine': ['refine_cloze_cn']
}

# Generate reasoning path or not, only for single choice
with_reasoning = False

# Use circular evaluation or not
with_circular_eval = True

mathbench_datasets = []

for _split in list(mathbench_sets.keys()):
    for _name in mathbench_sets[_split]:
        mathbench_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(
                            role='HUMAN',
                            prompt=single_choice_prompts[_name + '_with_reasoning'] if with_reasoning else single_choice_prompts[_name],
                        ),
                        dict(role='BOT', prompt='{answer}')] if 'choice' in _name else cloze_prompts[_name],
                    ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=512),
        )

        mathbench_eval_cfg = dict(
            evaluator=dict(type=CircularEvaluator if 'choice' in _name and with_circular_eval else AccEvaluator),
            pred_postprocessor=dict(type=first_option_postprocess, options='ABCD') if 'single_choice' in _name else dict(type=mathbench_postprocess, name=_name))

        mathbench_datasets.append(
            dict(
                abbr='mathbench-' + _split + '-' + _name,
                type=MathBenchDataset,
                path=f'./data/mathbench/{_split}',
                name=_name,
                with_circular=with_circular_eval,
                reader_cfg=dict(
                    input_columns=['question'],
                    output_column='answer'
                    ),
                infer_cfg=mathbench_infer_cfg,
                eval_cfg=mathbench_eval_cfg,
            ))
