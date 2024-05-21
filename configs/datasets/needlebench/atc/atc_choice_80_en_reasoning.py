from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import CircularEvaluator
from opencompass.datasets.needlebench.atc_choice import NeedleBenchATCDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

# ----------------------- Prompt Content----------------------- #

few_shot_prompts = {
    'single_choice_prompts': {
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
needle_num_list = list(range(2, 80, 1))
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
