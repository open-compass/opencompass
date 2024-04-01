from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import CompassArenaDataset

subjective_reader_cfg = dict(
    input_columns=['question', 'ref'],
    output_column='judge',
    )

data_path ="data/subjective/"

subjective_datasets = []

eng_base_prefix_score_with_ref = """
You are an assistant skilled at evaluating the quality of creative text.
Please evaluate the quality of an AI model's response to a creative question in the capacity of an impartial judge. You'll need to assess the response on the following dimensions: Creativity, Richness, User Demand Fulfillment, and Logical Coherence. We will provide you with a creative question and the AI model's response and a reference answer for your evaluation. As you begin your assessment, follow this process:
1. Evaluate the AI model's answers on different dimensions, pointing out its strengths or weaknesses in each dimension and assigning a score of 1 to 10 for each.
2. Finally, based on the assessments across dimensions, provide an overall score of 1 to 10 for the AI model's response.
3. Your scoring should be as stringent as possible and follow the scoring rules below: 

In general, the higher the quality of the model's response and its strict adherence to user needs, the higher the score. Responses that do not meet user needs will receive lower scores.

Scoring rules:
Creativity:
Scores 1-2 when there is no innovation or uniqueness in the content.
Scores 3-4 when providing partially original content but with low creative quality.
Scores 5-6 when mostly creative but lacks significant novelty, with moderate quality.
Scores 7-8 when having novelty and high-quality content.
Scores 9-10 when highly novel and of exceptional quality compared to the reference answer.

Richness:
Scores 1-2 when lacking depth and breadth, with very limited information.
Scores 3-4 when limited in depth and breadth, with fewer explanations and examples, showing low diversity.
Scores 5-6 when limited in depth and breadth but provides basic necessary information.
Scores 7-8 when providing depth and useful additional information.
Scores 9-10 when providing exceptional depth, breadth, and high diversity compared to the reference answer.

User Demand Fulfillment:
Scores 1-2 when completely irrelevant to the requirements, especially in style, theme, or significant word count difference.
Scores 3-4 when limited understanding of requirements, providing minimal relevant information, unable to help solve the problem, and significant discrepancies in style, theme, or word count.
Scores 5-6 when partially understanding requirements, providing somewhat relevant responses, with basic adherence to style, theme, and word count.
Scores 7-8 when understanding requirements well, providing highly relevant responses, adhering to style, theme, and word count requirements.
Scores 9-10 when accurately understanding all requirements, providing highly relevant and personalized responses, fully adhering to style, theme, and word count requirements, and significantly better meeting user needs than the reference answer.

Logical Coherence:
Scores 1-2 when entirely incoherent, lacking any logic, and not matching the question or known information.
Scores 3-4 when somewhat coherent but with many logical errors or inconsistencies.
Scores 5-6 when mostly coherent, with few errors, but may struggle to maintain complete coherence in complex situations.
Scores 7-8 when excellent logical handling, very few errors.
Scores 9-10 when flawless logic, impeccable in handling complexity, and significantly higher logical coherence compared to the reference answer.

Overall Score:
Scores 1-2 when irrelevant to the question, factually incorrect, or generates harmful content.
Scores 3-4 when no serious errors, mostly harmless, but of low quality and does not meet requirements.
Scores 5-6 when basically meeting requirements but performing poorly in some dimensions, with moderate quality.
Scores 7-8 when performing well in all dimensions.
Scores 9-10 when fully addressing user questions and all requirements, significantly surpassing the reference answer.

Please remember, you must evaluate and explain before scoring. After your explanation for each dimension, add the score for that dimension. Finally, at the end of your response, in the format of the dictionary (including brackets), return all your scoring results, ensuring your scores are integers:
{'Dimension One': Score, 'Dimension Two': Score, ..., 'Overall Score': Score}, for example: {'Creativity': 9, 'Richness': 6, ..., 'Overall Score': 7}.\n
"""


creation_prompt = """
请根据提供的 评分要求，用户问题 以及 相应的两个回答（回答1，回答2），判断两个回答中哪一个更好。
评分要求（重要性依次递减）:
1. 好的回答必须首先符合用户问题里的各种需求，不能跑题 
2. 好的回答必须具有逻辑连贯性，围绕一个中心进行回答
3. 好的回答必须具有创造性的词语和表达丰富度

[用户问题]
{question}
""" + base_prompt

sub_map = {"knowledge": knowledge_prompt, "language": language_prompt, "math_v2": math_prompt, "reason_v2": reason_prompt, "creationv2_zh": creation_prompt}

for _name, _prompt in sub_map.items():
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt="{question}"
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_seq_len=4096, max_out_len=2048),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            infer_order='double',
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt = _prompt
                    ),
                ]),
            ),
        ),
        pred_role="BOT",
    )

    subjective_datasets.append(
        dict(
            abbr=f"{_name}",
            type=CompassArenaDataset,
            path=data_path,
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg
        ))
