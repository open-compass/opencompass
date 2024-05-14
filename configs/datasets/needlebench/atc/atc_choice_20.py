from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import CircularEvaluator
from opencompass.datasets.needlebench.atc_choice import NeedleBenchATCDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

# ----------------------- Prompt Content----------------------- #

few_shot_prompts = {
    'single_choice_prompts': {
        'single_choice_cn': [
            dict(role='HUMAN', prompt='问题: 毕东作为刘红的爸爸，对刘红的成长有重要影响。 \n在上面提供的打乱的家族关系文本中，"刘红"的能够向上追溯到的最年长的亲人是谁？\nA. 毕东\nB. 刘红\nC. 李清亮\nD. 王展\n'),
            dict(role='BOT', prompt='回答: A'),
            dict(role='HUMAN', prompt='问题: 梅建平作为黄玉兰的姥姥，对黄玉兰的成长有重要影响。梅建平的妈妈是朱丽丽。蓝鑫把黄玉兰视为其母亲。焦慧不仅是朱丽丽的妈妈，还是朱丽丽的榜样。 \n在上面提供的打乱的家族关系文本中，"蓝鑫"的能够向上追溯到的最年长的亲人是谁？\nA. 梅建平\nB. 朱丽丽\nC. 蓝鑫\nD. 焦慧\n'),
            dict(role='BOT', prompt='回答: D'),
            dict(role='HUMAN', prompt='问题: 毕东把柳金凤视为其姥姥。奉兵作为柳金凤的妈妈，对柳金凤的成长有重要影响。余萍把杨颖视为其爸爸。毕东在郭建华的生命中扮演着父亲的角色。常宁的外公是余萍。刘慧是郭建华所生的孩子。刘慧在杨颖的生命中扮演着外公的角色。 \n在上面提供的打乱的家族关系文本中，"常宁"的能够向上追溯到的最年长的亲人是谁？\nA. 柳金凤\nB. 毕东\nC. 奉兵\nD. 余萍\n'),
            dict(role='BOT', prompt='回答: C'),
            dict(role='HUMAN', prompt='问题: 魏丽丽在谢平的生命中扮演着奶奶的角色。郭兵是魏阳的姥姥。谢平是郑玉珍的外婆。丁颖把武波视为其外公。丁颖在李建国的生命中扮演着外婆的角色。武波的父亲是刘瑜。许玲把余桂芳视为其父亲。刘瑜把许玲视为其爷爷。李建国对郭兵来说，不只是一个爷爷，还是一个朋友。魏丽丽的外公是魏阳。 \n在上面提供的打乱的家族关系文本中，"郑玉珍"的能够向上追溯到的最年长的亲人是谁？\nA. 魏丽丽\nB. 刘瑜\nC. 李建国\nD. 余桂芳\n'),
            dict(role='BOT', prompt='回答: D'),
            dict(role='HUMAN', prompt='问题: {question}'),
        ],
        'single_choice_en': [
            dict(role='HUMAN', prompt="Question: Jasmine Lane plays the role of James Hill's father in James Hill's life. \nGiven the scrambled family relationships described above, who is the eldest relative that 'James Hill' can trace back to in the context?\nA. Jasmine Lane\nB. James Hill\nC. Christopher Flores\nD. Paula Reeves\n"),
            dict(role='BOT', prompt='Answer: A'),
            dict(role='HUMAN', prompt="Question: For Michael Goodwin, Annette Valdez is not just a dad, but also a friend.For Annette Valdez, Mary Sanders is not just a father, but also a friend. \nGiven the scrambled family relationships described above, who is the eldest relative that 'Michael Goodwin' can trace back to in the context?\nA. Michael Goodwin\nB. Annette Valdez\nC. Mary Sanders\nD. James Mitchell\n"),
            dict(role='BOT', prompt='Answer: C'),
            dict(role='HUMAN', prompt="Question: Laura Garcia considers Mr. Stephen Valdez DDS as their mom.Mr. Stephen Valdez DDS's maternal grandfather is Elizabeth Smith.Laura Garcia is Ashley Johnson's mom. \nGiven the scrambled family relationships described above, who is the eldest relative that 'Ashley Johnson' can trace back to in the context?\nA. Ashley Johnson\nB. Mr. Stephen Valdez DDS\nC. Laura Garcia\nD. Elizabeth Smith\n"),
            dict(role='BOT', prompt='Answer: D'),
            dict(role='HUMAN', prompt="Question: Justin Sanchez's paternal grandfather is Laurie Bean.For Laurie Bean, Christina Kim is not just a mother, but also a friend.Jessica Bass, as Christina Kim's maternal grandfather, has a significant impact on Christina Kim's upbringing.Sarah Hampton considers Justin Sanchez as their father. \nGiven the scrambled family relationships described above, who is the eldest relative that 'Sarah Hampton' can trace back to in the context?\nA. Christina Kim\nB. Jessica Bass\nC. Laurie Bean\nD. Justin Sanchez\n"),
            dict(role='BOT', prompt='Answer: B'),
            dict(role='HUMAN', prompt='Question: {question}'),
        ],
        'single_choice_cn_reasoning': [
            dict(role='HUMAN', prompt='问题: 毕东作为刘红的爸爸，对刘红的成长有重要影响。 \n在上面提供的打乱的家族关系文本中，"刘红"的能够向上追溯到的最年长的亲人是谁？\nA. 毕东\nB. 刘红\nC. 李清亮\nD. 王展\n'),
            dict(role='BOT', prompt='回答: 根据提供的信息，我们可以构建以下家族关系链：\n\n- 刘红的爸爸是毕东。\n\n因此，"刘红"能够向上追溯到的最年长的亲人是毕东。所以答案是：A'),

            dict(role='HUMAN', prompt='问题: 梅建平作为黄玉兰的姥姥，对黄玉兰的成长有重要影响。梅建平的妈妈是朱丽丽。蓝鑫把黄玉兰视为其母亲。焦慧不仅是朱丽丽的妈妈，还是朱丽丽的榜样。 \n在上面提供的打乱的家族关系文本中，"蓝鑫"的能够向上追溯到的最年长的亲人是谁？\nA. 梅建平\nB. 朱丽丽\nC. 蓝鑫\nD. 焦慧\n'),
            dict(role='BOT', prompt='回答: 根据提供的信息，我们可以构建以下家族关系链：\n\n- 蓝鑫把黄玉兰视为其母亲。\n- 黄玉兰的姥姥是梅建平。\n- 梅建平的妈妈是朱丽丽。\n- 朱丽丽的妈妈是焦慧。\n\n因此，"蓝鑫"能够向上追溯到的最年长的亲人是焦慧。所以答案是：D'),

            dict(role='HUMAN', prompt='问题: 毕东把柳金凤视为其姥姥。奉兵作为柳金凤的妈妈，对柳金凤的成长有重要影响。余萍把杨颖视为其爸爸。毕东在郭建华的生命中扮演着父亲的角色。常宁的外公是余萍。刘慧是郭建华所生的孩子。刘慧在杨颖的生命中扮演着外公的角色。 \n在上面提供的打乱的家族关系文本中，"常宁"的能够向上追溯到的最年长的亲人是谁？\nA. 柳金凤\nB. 毕东\nC. 奉兵\nD. 余萍\n'),
            dict(role='BOT', prompt='回答: 根据提供的信息，我们可以构建以下家族关系链：\n\n- 常宁的外公是余萍。\n- 余萍把杨颖视为其爸爸。\n- 杨颖的外公是刘慧。\n- 刘慧是郭建华所生的孩子。\n- 郭建华的父亲是毕东。\n- 毕东的姥姥是柳金凤。\n- 柳金凤的妈妈是奉兵。\n\n因此，"常宁"能够向上追溯到的最年长的亲人是奉兵。所以答案是：C'),

            dict(role='HUMAN', prompt='问题: 魏丽丽在谢平的生命中扮演着w奶奶的角色。郭兵是魏阳的姥姥。谢平是郑玉珍的外婆。丁颖把武波视为其外公。丁颖在李建国的生命中扮演着外婆的角色。武波的父亲是刘瑜。许玲把余桂芳视为其父亲。刘瑜把许玲视为其爷爷。李建国对郭兵来说，不只是一个爷爷，还是一个朋友。魏丽丽的外公是魏阳。 \n在上面提供的打乱的家族关系文本中，"郑玉珍"的能够向上追溯到的最年长的亲人是谁？\nA. 魏丽丽\nB. 刘瑜\nC. 李建国\nD. 余桂芳\n'),
            dict(role='BOT', prompt='回答: 根据提供的信息，我们可以构建以下家族关系链：\n\n- 郑玉珍的外婆是谢平。\n- 谢平的奶奶是魏丽丽。\n- 魏丽丽的外公是魏阳。 \n- 魏阳的姥姥是郭兵。\n- 郭兵的爷爷是李建国。 \n- 李建国的外婆是丁颖。 \n- 丁颖的外公是武波。 \n- 武波的父亲是刘瑜。\n- 刘瑜的爷爷是许玲。\n- 许玲的父亲是余桂芳。 \n\n因此，"郑玉珍"能够向上追溯到的最年长的亲人是余桂芳。所以答案是：D'),
            dict(role='HUMAN', prompt='问题: {question}'),
        ],
        'single_choice_en_reasoning': [
            dict(role='HUMAN', prompt="Question: Sharon House, as Jessica Stewart's father, has a significant impact on Jessica Stewart's upbringing. \nGiven the scrambled family relationships described above, who is the eldest relative that 'Jessica Stewart' can trace back to in the context?\nA. Jack Burch\nB. Jessica Stewart\nC. Sharon House\nD. Carolyn Jackson\n"),
            dict(role='BOT', prompt="Answer: Based on the provided information, we can construct the following family relationship chain：\n\n- Sharon House, as Jessica Stewart's father, has a significant impact on Jessica Stewart's upbringing.\n\nTherefore, the eldest relative that 'Jessica Stewart' can trace back to in the context is Sharon House. The answer is: C"),
            dict(role='HUMAN', prompt="Question: For Robert Hill, Mikayla Scott is not just a paternal grandfather, but also a friend.Jacob Oconnor's paternal grandmother is Robert Hill. \nGiven the scrambled family relationships described above, who is the eldest relative that 'Jacob Oconnor' can trace back to in the context?\nA. Laura Holland\nB. Robert Hill\nC. Jacob Oconnor\nD. Mikayla Scott\n"),
            dict(role='BOT', prompt="Answer: Based on the provided information, we can construct the following family relationship chain：\n\n-Jacob Oconnor's paternal grandmother is Robert Hill. \n- For Robert Hill, Mikayla Scott is not just a paternal grandfather, but also a friend.\n\nTherefore, the eldest relative that 'Jacob Oconnor' can trace back to in the context is Mikayla Scott. The answer is: D"),
            dict(role='HUMAN', prompt="Question: Misty Moore plays the role of Barbara Fuentes's maternal grandfather in Barbara Fuentes's life.Jennifer Garcia, as Michael Martinez's grandmother, has a significant impact on Michael Martinez's upbringing.Michael Martinez is not only Misty Moore's father but also Misty Moore's role model. \nGiven the scrambled family relationships described above, who is the eldest relative that 'Barbara Fuentes' can trace back to in the context?\nA. Michael Martinez\nB. Jennifer Garcia\nC. Misty Moore\nD. Barbara Fuentes\n"),
            dict(role='BOT', prompt="Answer: Based on the provided information, we can construct the following family relationship chain：\n- Misty Moore plays the role of Barbara Fuentes's maternal grandfather in Barbara Fuentes's life. \n- Michael Martinez is not only Misty Moore's father but also Misty Moore's role model.\n- Jennifer Garcia, as Michael Martinez's grandmother, has a significant impact on Michael Martinez's upbringing.\n\nTherefore, the eldest relative that 'Barbara Fuentes' can trace back to in the context is Jennifer Garcia. The answer is: B"),
            dict(role='HUMAN', prompt="Question: Carlos Smith, as Mary Gay's grandfather, has a significant impact on Mary Gay's upbringing.Victor Dean considers Mary Gay as their grandfather.Marcus Miller, as Carlos Smith's paternal grandfather, has a significant impact on Carlos Smith's upbringing.Victor Dean is not only Danielle Yates's maternal grandmother but also Danielle Yates's role model.Danielle Yates is not only David Hernandez's paternal grandmother but also David Hernandez's role model.David Hernandez is Jennifer Williams's mom. \nGiven the scrambled family relationships described above, who is the eldest relative that 'Jennifer Williams' can trace back to in the context?\nA. Marcus Miller\nB. Carlos Smith\nC. Mary Gay\nD. Victor Dean\n"),
            dict(role='BOT', prompt="Answer: Based on the provided information, we can construct the following family relationship chain：\n\n- David Hernandez is Jennifer Williams's mom.\n- Danielle Yates is not only David Hernandez's paternal grandmother but also David Hernandez's role model.\n- Victor Dean is not only Danielle Yates's maternal grandmother but also Danielle Yates's role model.\n- Victor Dean considers Mary Gay as their grandfather. \n- Carlos Smith, as Mary Gay's grandfather, has a significant impact on Mary Gay's upbringing.\n- Marcus Miller, as Carlos Smith's paternal grandfather, has a significant impact on Carlos Smith's upbringing.\n\nTherefore, the eldest relative that 'Jennifer Williams' can trace back to in the context is Marcus Miller. The answer is: A"),
            dict(role='HUMAN', prompt='Question: {question}'),
        ],
    },
}

# ----------------------- Prompt Settings ----------------------- #
needle_num_list = list(range(2, 20, 1))
names_path = './data/needlebench/names.json'

repeats = 10

# Use Zero-Shot or not
with_few_shot = True

# Max for this dataset is 4, should be set with `with_few_shot`
few_shot_samples = 4

# Generate reasoning path or not, only for single choice
with_reasoning = True

# Use circular evaluation or not
with_circular_eval = True

needlebench_prompts = few_shot_prompts
single_choice_prompts = needlebench_prompts['single_choice_prompts']

# Set few shot prompt number
for _name in list(single_choice_prompts.keys()):
    if with_few_shot:
        assert few_shot_samples > 0 and few_shot_samples <= 4
        single_choice_prompts[_name] = \
            single_choice_prompts[_name][- few_shot_samples * 2 - 1:]

# ----------------------- Dataset Settings ----------------------- #

needlebench_datasets = []


needlebench_atc_reader_cfg = dict(input_columns=['question'],
                                  output_column='answer')

for _name in list(single_choice_prompts.keys()):

    needlebench_atc_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=(single_choice_prompts[_name])),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer,),
    )

    needlebench_atc_eval_cfg = dict(
        evaluator=dict(type=CircularEvaluator),
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

    for num_needles in needle_num_list:
        abbr = (f'NeedleBenchATCDataset-'
                f'{num_needles}Needle-{"EN" if "en" in _name else "ZH"}')
        language = 'English' if 'en' in _name else 'Chinese'
        if 'reasoning' in _name:
            abbr += '-Reasoning'
        dataset_dict = {
            'abbr': abbr,
            'type': NeedleBenchATCDataset,
            'path': names_path,
            'num_needles': num_needles,
            'language': language,
            'repeats': repeats,
            'with_circular': with_circular_eval,
            'reader_cfg': needlebench_atc_reader_cfg,
            'infer_cfg': needlebench_atc_infer_cfg,
            'eval_cfg': needlebench_atc_eval_cfg
        }
        needlebench_datasets.append(dataset_dict)
