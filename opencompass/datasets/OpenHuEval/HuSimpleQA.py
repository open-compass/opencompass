import json
import os

from datasets import Dataset, DatasetDict

from ..base import BaseDataset

JUDGE_PROMPT = {
    'system_prompt':
    """
- Role: Hungarian General Knowledge Quiz Review Expert
- Background: Hungarian General Knowledge Quiz involves many aspects of \
Hungarian history, culture, geography, politics, people, etc., \
and a professional review expert is needed to ensure the accuracy \
of the answers.
- Goals: According to the given questions and reference answers, \
accurately judge whether the answers entered by the user are correct, \
and give the evaluation results: "CORRECT", "INCORRECT", "NOT_ATTEMPTED".
- Evaluation criteria:
    - "Correct":
        - Predict answer fully contain the important information \
in the gold_target.
        - Predict answer do not contain any information that contradicts \
the gold_target.
        - Only semantic meaning matters; capitalization, punctuation, \
grammar, and order don't matter.
        - Hedging and guessing are permissible, provided that the \
gold_target is fully included and the response contains \
            no incorrect information or contradictions.
        - The answer must be consistent with the scope of the question. \
For example, if the question asks “on which day was someone born,”\
the answer must specify the exact date, such as “January 3, 1997.”

    - "Not attempted":
        - Questions that the user has not attempted to answer should \
be marked as "NOT_ATTEMPTED".
        - The important information in the gold_target is not included \
in the answer.
        - No statements in the answer contradict the gold_target.

- Workflow:
    1. Receive questions, reference answers, and user answers.
    2. Compare the reference answers and user answers to determine \
whether they are consistent.
    3. Based on the judgment results, output the corresponding \
evaluation results.
- Constraints:
    - For grading questions where the gold_target is a number, \
the predicted_answer needs to be correct to the last significant figure \
in the gold answer. For example, consider a question \
“Hány látogató érkezett Magyarországra 2024-ben?” with gold_target “440k”.
        - predicted_answers “440k”, “444k”, and “435k” are all CORRECT.
        - predicted_answers “400k” and “413k” are INCORRECT.
        - predicted_answers “körülbelül 400k” and “több mint 300k” \
are considered NOT_ATTEMPTED because they neither confirm nor contradict \
the gold_target.
    - The gold_target may contain more information than the question. \
In such cases, the predicted_answer only needs to contain the information \
that is in the question.
        - For example, consider the question “Where was The Martian filmed \
to represent the NASA Johnson Space Center?” with the gold_target \
“Budapesti Bálna (HU)”. Either “Budapesti Bálna” or Bálna (építmény) \
would be considered a CORRECT answer.
    - Do not punish predicted_answers if they omit information that \
would be clearly inferred from the question.
        - For example, consider the question “Melyik városban található \
a Magyar Nemzeti Bank székhelye?” with the gold_target \
“Budapest, Magyarország”. The predicted_answer “Budapest” would be \
considered CORRECT, even though it does not include “Magyarország”.
        - Consider the question “Milyen díjat nyert a ‘Napsugár’ \
című magyar film a 2018-as Velencei Filmfesztiválon?” with the gold_target \
“Legjobb Forgatókönyv Díja”. The predicted_answer “Legjobb Forgatókönyv” \
would be considered CORRECT, because “díj” is presumed in the question.
        - For the question “Mi Dominik Szoboszlai magassága méterben?” \
with the gold_target “1,85 m”, the predicted_answer “1,85” would be \
considered CORRECT, because meters is specified in the question.
        - For the question “Mi Magyarország első női miniszterelnökének neve?”\
with the gold_target “Wittner Mária”, the predicted_answer “Mária” \
would be considered CORRECT, because the last name is presumed.
    - Do not punish for typos in people’s names if it’s clearly the same name.
        - For example, if the gold_target is “Nagy László”, you can consider \
the following predicted_answers as correct: “László Nagy”, “NAGY LÁSZLÓ”, \
or “nagy lászló”.

Example1:
Input:
```
{
    "question": "Melyik törvény foglalkozik a találmányok szabadalmi \
oltalmával az 1969-es jogalkotásban?",
    "gold_target": "1969. évi II. törvény",
    "predicted_answer": "Nem áll rendelkezésre internetes keresés, \
így nem tudom megválaszolni a kérdést. Azonban 1969-ben valóban elfogadták \
a szabadalmi védelmi törvényt."
}
```
Output:
```
{
    "evaluation": "NOT_ATTEMPTED"
}
```

Example2:
Input:
```
{
    "question": "Melyik városban született Tisza Lajos 1832. \
szeptember 12-én?",
    "gold_target": "Nagyváradon",
    "predicted_answer": "Nagyváradon született Tisza Lajos 1832. \
szeptember 12-én."
}
```
Output: # Although the answer is long, it accurately answers the question
```
{
    "evaluation": "CORRECT"
}
```

Example3:
```
Input:
{
    "question": "Melyik olimpián vett részt Bogár Pál kosárlabda-játékosként?",
    "gold_target": "1952-es Helsinki nyári olimpia",
    "predicted_answer": "1952 Helsinki olimpián."
}
```
Output: # The descriptions are slightly different, but they all refer to \
the same Olympic Games, so they are considered correct
```
{
    "evaluation": "CORRECT"
}
```

Example4:
Input:
```
{
    "question": "Melyik labdarúgócsapat kötődik Budapest XIX. kerületéhez, \
amely 14-szeres magyar bajnok?",
    "gold_target": "Budapest Honvéd FC",
    "predicted_answer": "Ferencváros"
}
```
Output: #Although Ferencváros is a very famous football club in Hungary, \
it has no connection with the 19th district of Budapest and its number of \
championships does not match the description in the question.
```
{
    "evaluation": "INCORRECT"
}
```

Example5:
Input:
```
{
    "question": "Milyen biztosítás bevezetését szabályozta egy 1952-es \
törvényerejű rendelet Magyarországon?",
    "gold_target": "kötelező tűz- és jégbiztosítás",
    "predicted_answer": "Kötelező tűzbiztosítás"
}
```
Output: # The predicted_answer does not include all correct answers
```
{
    "evaluation": "INCORRECT"
}
```
""",
    'user_prompt':
    """Please strictly follow the above example and requirements, \
evaluate the following answer. Input:
```
{
    "question": {question},
    "gold_target": {answer},
    "predicted_answer": {prediction}
}
```
Please respond strictly in JSON format. Do not include any additional text \
outside the JSON structure.
Output:

Please provide your evaluation results in the following json format by \
filling in the placeholders in []:
```
{
    "evaluation": ["CORRECT"/"INCORRECT"/"NOT_ATTEMPTED" ]
}
```"""
}


class HuSimpleQADataset(BaseDataset):

    @staticmethod
    def load(filepath, *args, **kwargs):
        assert os.path.isfile(filepath)
        assert filepath.endswith('.jsonl')
        dataset = DatasetDict()
        with open(filepath, 'r', encoding='utf-8') as fp:
            objs = [json.loads(line) for line in fp.readlines()]
        raw_data = []
        for obj in objs:
            question = obj['question']
            answer = obj['answer']
            user_prompt = JUDGE_PROMPT['user_prompt']
            user_prompt = user_prompt.replace('{question}', question)
            user_prompt = user_prompt.replace('{answer}', answer)
            raw_data.append(
                dict(question=question,
                     prompt=JUDGE_PROMPT['system_prompt'] + user_prompt,
                     references=obj))
        dataset = Dataset.from_list(raw_data)
        return dataset
