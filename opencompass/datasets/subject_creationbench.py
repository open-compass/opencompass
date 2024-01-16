# flake8: noqa: E501
import json
import os.path as osp
import re
from typing import Optional

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .subjective_cmp import SubjectiveCmpDataset

eng_base_prefix = """
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
{'Dimension One': Score, 'Dimension Two': Score, ..., 'Overall Score': Score}, for example: {'Creativity': 9, 'Richness': 6, ..., 'Overall Score': 7}.\n
"""

chn_base_prefix = """
你是一个擅长评价创作类文本质量的助手。
请你以公正的评判者的身份，评估一个AI模型对创作类问题的回答的质量。你需要从下面的几个维度对回答进行评估：创造性，丰富度，满足用户需求，逻辑连贯性
我们会给您提供一个创作类问题，和需要你评估的AI模型的回答。当你开始你的评估时，你需要遵守以下的流程：
1. 从不同维度对AI模型的答案进行评价，指出AI模型的答案有哪些优点或不足，在每个维度的评价之后，给每一个维度一个1～10的分数。
2. 最后，综合每个维度的评估，对AI模型的回答给出一个1～10的综合得分。
3. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。

创造性评分规则：
当模型的回答没有能够提供任何创新性或独特性内容时，创造性得分必须是1到2分；
当模型的回答能够提供部分原创性的创作内容，但创作质量较低时，创造性得分为3到4分；
当模型的回答基本均为创造性内容，但在创作上无太多新意，质量中等，创造性得分可以得5到6分；
当模型的回答具有新意，且创作内容质量较高时，创造性得分得到7到8分；
当模型的回答的创作内容非常新颖且质量极高时，创造性得分才能得到9到10分。

丰富度评分规则：
当模型的回答很不丰富，缺乏深度和广度，提供的信息非常有限，信息呈现出很低的多样性时，丰富度得分必须是1到2分；
当模型的回答较不丰富，缺乏必要的深度，解释和实例较少，且可能不够相关或不够详细，上下文考虑有限，信息展现出较低的多样性时，丰富度得分为3到4分；
当模型的回答较为丰富，但深度和广度有限，信息多样性一般，用户能够从回答中获得基本所需的信息时，丰富度得分可以得5到6分；
当模型的回答丰富，并提供了一定的深度，上下文考虑较为全面，信息展现出一定的多样性，用户能够从回答中获得所需以及一些额外的有用信息时，丰富度得分得到7到8分；
当模型的回答非常丰富，提供了额外的深度和广度，包含多个相关的详细解释和实例以增强理解，上下文考虑全面，信息呈现出高度的多样性时，丰富度得分才能得到9到10分。

满足用户需求评分规则：
当模型的回答与用户需求完全不相关，无法满足用户的基本需求，特别是文体、主题完全不符，字数要求相差很大时，满足用户需求得分必须是1到2分；
当模型的回答对用户需求的理解有限，只能在很小程度上提供相关信息，与用户需求关联性较低，不太能够帮助用户解决问题，文体、主题、字数与题目要求相差较大时，满足用户需求得分为3到4分；
当模型的回答能够部分理解用户需求，并提供部分相关的解决方案或回应，文体、主题基本符合需求，字数与要求相差不大时，满足用户需求得分可以得5到6分；
当模型的回答能够较好地理解用户需求，并提供较为相关的解决方案或回应，文体、主题、字数符合问题要求时，满足用户需求得分得到7到8分；
当模型的回答能够精准地理解用户的所有需求，并提供高度相关和个性化的解决方案或回应，文体、主题、字数完全符合用户需求时，满足用户需求得分才能得到9到10分。

逻辑连贯性评分规则：
当模型的回答完全不连贯，没有任何逻辑性，与问题或已知信息完全不匹配时，逻辑连贯性得分必须是1到2分；
当模型的回答在一定程度上是逻辑连贯的，但仍有不少逻辑错误或不一致之处时，逻辑连贯性得分为3到4分；
当模型的回答在大多数情况下是逻辑连贯的，逻辑错误较少，但在某些复杂情况下可能无法保持完全的连贯性时，逻辑连贯性得分可以得5到6分；
当模型的回答在逻辑连贯性方面表现出色，能够很好地处理复杂逻辑，错误非常少见，且能够处理复杂的逻辑任务时，逻辑连贯性得分得到7到8分；
当模型的回答在逻辑连贯性方面达到完美，无论问题多么复杂或具有挑战性，都能够展现出无懈可击的逻辑能力，没有任何逻辑错误时，逻辑连贯性得分才能得到9到10分。

综合得分评分规则：
当模型回答存在与问题不相关，或者有本质性的事实错误，或生成了有害内容时，总分必须是1到2分；
当模型回答没有严重错误而且基本无害，但是质量较低，没有满足用户需求，总分为3到4分；
当模型回答基本满足用户要求，但是在部分维度上表现较差，质量中等，总分可以得5到6分；
当模型回答在所有维度上表现良好，总分得7到8分；
只有当模型回答充分地解决了用户问题和所有需求，并且在所有维度上都接近满分的情况下，才能得9到10分。

请记住，你必须在你打分前进行评价和解释。在你对每个维度的解释之后，需要加上对该维度的打分。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整数：
{'维度一': 打分, '维度二': 打分, ..., '综合得分': 打分}，例如：{'创造性': 9, '丰富度': 6, ..., '综合得分': 7}。\n
"""

chn_base_prefix_score_with_ref = """
你是一个擅长评价创作类文本质量的助手。
请你以公正的评判者的身份，评估一个AI模型对创作类问题的回答的质量。你需要从下面的几个维度对回答进行评估：创造性，丰富度，满足用户需求，逻辑连贯性
我们会给您提供一个创作类问题，和需要你评估的AI模型的回答以及一个参考答案。当你开始你的评估时，你需要遵守以下的流程：
1. 从不同维度对AI模型的答案进行评价，给每一个维度一个1～10的分数。
2. 最后，综合每个维度的评估，对AI模型的回答给出一个1～10的综合得分。
3. 你的打分需要尽可能严格，并且要遵守下面的评分规则：

总的来说，模型回答的质量越高，且严格符合用户的需求，则分数越高，不符合用户需求的回答分数越低。
评分规则：
创造性：
- 未能够提供任何创新性或独特性内容时，得1-2分；
- 提供部分原创性的创作内容，但创作质量较低时，得3-4分；
- 基本均为创造性内容，但无太多新意，质量中等，得5-6分；
- 具有新意，且内容质量较高时，得7-8分；
- 非常新颖且质量极高时，相比参考答案明显创造性质量更高，得9-10分。
丰富度：
- 缺乏深度和广度，信息量非常有限，得1-2分；
- 缺乏必要的深度，解释和实例较少，不够相关或不够详细，信息展现出较低的多样性时，得3-4分；
- 但深度和广度有限，信息多样性一般，仅能从中获得基本所需的信息时，得5-6分；
- 提供一定的深度，可从中获得必要及额外的有用信息时，得7-8分；
- 提供了额外的深度和广度，包含多个相关的详细解释和实例，多样性很高，相比参考答案明显丰富度更高，得9-10分。
满足用户需求：
- 与需求完全不相关，特别是文体、主题完全不符，字数要求相差很大时，得1-2分；
- 对需求的理解有限，只能在很小程度上提供相关信息，不太能够帮助用户解决问题，文体、主题、字数与题目要求相差较大时，得3-4分；
- 部分理解需求，提供部分相关的回应，文体、主题基本符合需求，字数与要求相差不大时，得5-6分；
- 较好地理解需求，提供较为相关的回应，文体、主题、字数符合问题要求时，得7-8分；
- 精准地理解所有需求，提供高度相关和个性化的回应，文体、主题、字数完全符合需求时，相比参考答案更能满足用户需求，得9-10分。
逻辑连贯性：
- 完全不连贯，没有任何逻辑性，与问题或已知信息完全不匹配时，得1-2分；
- 一定程度上逻辑连贯，但仍有不少逻辑错误或不一致之处时，得3-4分；
- 大多数情况下逻辑连贯，错误较少，但在复杂情况下可能无法保持完全的连贯性时，得5-6分；
- 逻辑出色，很好地处理复杂逻辑，错误非常少见，得7-8分；
- 逻辑完美，无论问题多么复杂都有无懈可击的逻辑能力，相比参考答案明显逻辑连贯性更高，得9-10分。
综合得分：
- 存在与问题不相关，或事实错误，或生成有害内容时，得1-2分；
- 没有严重错误，基本无害，但质量较低，没有满足要求，得3-4分；
- 基本满足要求，但是在上述部分维度上表现较差，质量中等，得5-6分；
- 在所有维度上表现良好，得7到8分；
- 充分地解决了用户问题和所有需求，综合对比显著超过参考答案的情况下，得9-10分。

请记住，你必须在你打分前进行评价和解释。在你对每个维度的解释之后，需要加上对该维度的打分。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整数：
{'维度一': 打分, '维度二': 打分, ..., '综合得分': 打分}，例如：{'创造性': 9, '丰富度': 6, ..., '综合得分': 7}。\n
"""

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

compare_cn_prefix = """
请根据提供的 评分要求，用户问题 以及 相应的两个回答（回答1，回答2），判断两个回答中哪一个更好。
评分要求（重要性依次递减）:
1. 好的回答必须首先符合用户问题里的各种需求，不能跑题
2. 好的回答必须具有逻辑连贯性，围绕一个中心进行回答
3. 好的回答必须具有创造性的词语和表达丰富度

[用户问题]
"""

compare_cn_suffix = """
根据评分要求，在以下 2 个选项中做出选择:
A. 回答1更好
B. 回答2更好
并提供你的解释原因。

如果你认为回答1更好，你的输出应形如：
选择：A
原因：blahblah blahblah\n

如果你认为回答2更好，你的输出应形如：
选择：B
原因：blahblah blahblah\n
"""

compare_cn_prefix_4opt = """
请根据提供的 评分要求，用户问题 以及 相应的两个回答（回答1，回答2），判断两个回答中哪一个更好。
评分要求（重要性依次递减）:
1. 好的回答必须首先符合用户问题里的各种需求，不能跑题
2. 好的回答必须具有逻辑连贯性，围绕一个中心进行回答
3. 好的回答必须具有创造性的词语和表达丰富度

[用户问题]
"""

compare_cn_suffix_4opt = """
根据评分要求，在以下 2 个选项中做出选择:
A. 回答1更好
B. 回答2更好
C. 回答1、2都好
D. 回答1、2都不好
并提供你的解释原因。

如果你认为回答1更好，你的输出应形如：
选择：A
原因：blahblah blahblah\n

如果你认为回答2更好，你的输出应形如：
选择：B
原因：blahblah blahblah\n

如果你认为回答1、2都很好，你的输出应形如：
选择：C
原因：blahblah blahblah\n

如果你认为回答1、2都不好，你的输出应形如：
选择：D
原因：blahblah blahblah\n
"""


def prompt_construct(sample):
    lan = sample['others']['language']
    question = sample['question']
    if lan == 'zh':
        prompt = chn_base_prefix + '创作类问题：' + str(question) + '\n[模型回答开始]\n'
        suffix = '\n[模型回答结束]\n'
    elif lan == 'en':
        prompt = eng_base_prefix + 'Creative Question: ' + str(
            question) + "\n[Model's response start]\n"
        suffix = "\n[Model's response end]\n"
    return prompt, suffix


def prompt_construct_score_with_ref(sample):
    lan = sample['others']['language']
    question = sample['question']
    ref = sample['ref']
    if lan == 'zh':
        prompt = chn_base_prefix_score_with_ref + '创作类问题：' + str(
            question) + '\n[参考答案开始]\n' + str(
                ref) + '\n[参考答案结束]\n' + '\n[模型回答开始]\n'
        suffix = '\n[模型回答结束]\n'
    elif lan == 'en':
        prompt = eng_base_prefix_score_with_ref + 'Creative Question: ' + str(
            question) + '\n[Reference start]\n' + str(
                ref) + '\n[Reference end]\n' + "\n[Model's response start]\n"
        suffix = "\n[Model's response end]\n"
    return prompt, suffix


def prompt_construct_compare(sample):
    lan = sample['others']['language']
    question = sample['question']
    if lan == 'zh':
        prompt = compare_cn_prefix + str(question)
        suffix = compare_cn_suffix
    return prompt, suffix


def prompt_construct_compare_4opt(sample):
    lan = sample['others']['language']
    question = sample['question']
    if lan == 'zh':
        prompt = compare_cn_prefix_4opt + str(question)
        suffix = compare_cn_suffix_4opt
    return prompt, suffix


@LOAD_DATASET.register_module()
class CreationBenchDataset(SubjectiveCmpDataset):

    def load(self,
             path: str,
             name: str,
             multi_dimension: Optional[bool] = False):
        dataset = list(super().load(path, name))
        creation_dataset = []
        for data in dataset:
            if multi_dimension:
                prefix, suffix = prompt_construct(data)
                data['gpt4_prefix'] = prefix
                data['gpt4_suffix'] = suffix
            data['judge']['others'] = data['others']
            data['ref'] = data['others']['reference']
            data['score_with_ref_prefix'], data[
                'score_with_ref_suffix'] = prompt_construct_score_with_ref(
                    data)
            # data['compare_cn_prefix'], data['compare_cn_suffix'] = prompt_construct_compare(data)
            # data['compare_cn_prefix_4opt'], data['compare_cn_suffix_4opt'] = prompt_construct_compare_4opt(data)
            creation_dataset.append(data)
        dataset = Dataset.from_list(creation_dataset)
        return dataset
