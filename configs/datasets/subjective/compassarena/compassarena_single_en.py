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

creation_prompt = """
You are an assistant skilled at evaluating the quality of creative text.
Please evaluate the quality of an AI model's response to a creative question in the capacity of an impartial judge. You'll need to assess the response on the following dimensions: Creativity, Richness, User Demand Fulfillment, and Logical Coherence. We will provide you with a creative question and the AI model's response for evaluation. As you begin your assessment, follow this process:
1. Evaluate the AI model's answers on different dimensions, pointing out its strengths or weaknesses in each dimension and assigning a score of 1 to 10 for each.
2. Finally, based on the assessments across dimensions, provide an overall score of 1 to 10 for the AI model's response.
3. Your scoring should be as stringent as possible and follow the scoring rules below: 

Generally, the higher the quality of the model's response, the higher the score.

Creativity Scoring Guidelines:
When the model's response fails to provide any innovative or unique content, the creativity score must be between 1 and 2;
When the model's response partially offers original creative content but of low quality, the creativity score is between 3 and 4;
When the model's response consists mostly of creative content but lacks significant novelty in the creation, with average quality, the creativity score can range from 5 to 6;
When the model's response presents novelty and high-quality creative content, the creativity score ranges from 7 to 8;
When the model's response contains highly innovative and high-quality creative content, the creativity score can only reach 9 to 10.

Richness Scoring Guidelines:
When the model's response lacks richness, lacks depth and breadth, offers extremely limited information, and displays very low diversity in information, the richness score must be between 1 and 2;
When the model's response is somewhat lacking in richness, lacks necessary depth, explanations, and examples, might be less relevant or detailed, and has limited contextual considerations, the richness score is between 3 and 4;
When the model's response is somewhat rich but with limited depth and breadth, moderately diverse information, providing users with the necessary information, the richness score can range from 5 to 6;
When the model's response is rich, and provides some depth, comprehensive contextual considerations, and displays some diversity in information, the richness score ranges from 7 to 8;
When the model's response is extremely rich, offers additional depth and breadth, includes multiple relevant detailed explanations and examples to enhance understanding, comprehensive contextual considerations, and presents highly diverse information, the richness score can only reach 9 to 10.

User Demand Fulfillment Scoring Guidelines:
When the model's response is entirely unrelated to user demands, fails to meet basic user requirements, especially in style, theme, and significant word count differences, the user demand fulfillment score must be between 1 and 2;
When the model's response has limited understanding of user demands, only provides somewhat relevant information, lacks strong connections to user demands, unable to significantly aid in problem-solving, significant style, theme, and word count differences, the user demand fulfillment score is between 3 and 4;
When the model's response partially understands user demands, provides some relevant solutions or responses, the style, theme are generally in line with requirements, and the word count differences are not significant, the user demand fulfillment score can range from 5 to 6;
When the model's response understands user demands fairly well, offers fairly relevant solutions or responses, style, theme, and word count align with problem requirements, the user demand fulfillment score ranges from 7 to 8;
When the model's response accurately understands all user demands, provides highly relevant and personalized solutions or responses, style, theme, and word count entirely align with user requirements, the user demand fulfillment score can only reach 9 to 10.

Logical Coherence Scoring Guidelines:
When the model's response lacks any coherence, lacks any logical sequence, entirely mismatched with the question or known information, the logical coherence score must be between 1 and 2;
When the model's response is somewhat coherent but still has numerous logical errors or inconsistencies, the logical coherence score is between 3 and 4;
When the model's response is mostly coherent, with few logical errors, might lose coherence in certain complex situations, the logical coherence score can range from 5 to 6;
When the model's response excels in logical coherence, handles complex logic well, very few errors, can handle intricate logical tasks, the logical coherence score ranges from 7 to 8;
When the model's response achieves perfect logical coherence, flawless in handling complex or challenging questions, without any logical errors, the logical coherence score can only reach 9 to 10.

Overall Scoring Guidelines:
When the model's response is entirely irrelevant to the question, contains substantial factual errors, or generates harmful content, the overall score must be between 1 and 2;
When the model's response lacks severe errors and is generally harmless but of low quality, fails to meet user demands, the overall score ranges from 3 to 4;
When the model's response mostly meets user requirements but performs poorly in some dimensions, with average quality, the overall score can range from 5 to 6;
When the model's response performs well across all dimensions, the overall score ranges from 7 to 8;
Only when the model's response fully addresses user problems and all demands, achieving near-perfect scores across all dimensions, can the overall score reach 9 to 10.

Please remember, you must evaluate and explain before scoring. After your explanation for each dimension, add the score for that dimension. Finally, at the end of your response, in the format of the dictionary (including brackets), return all your scoring results, ensuring your scores are integers:
{'Dimension One': Score, 'Dimension Two': Score, ..., 'Overall Score': Score}, for example: {'Creativity': 9, 'Richness': 6, ..., 'Overall Score': 7}.

Creative Question:
{question}

[Model's response start]
{prediction}
[Model's response end]
"""

sub_map = {"creationv2_en": creation_prompt}

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
