import ast
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path
from opencompass.utils.prompt import PromptList

from .base import BaseDataset

OWNER_PROMPT_TEMPLATE = """You are participating in a puzzle solving competition. You are an expert at solving puzzles.
Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.

Respond in the format of the training output examples

--Training Examples--

{training_examples}

--End of Training Examples--

--Test Input--

{test_input}

--End of Test Input--

Your response:"""

SECOND_PASS_EXTRACTION_PROMPT = """You are a helpful assistant.
Extract only the JSON array of arrays from the following response.
Do not include any explanation, formatting, or additional text.
Return ONLY the valid JSON array of arrays with integers.

Response:
{response}

Example of expected output format:
[[1, 2, 3], [4, 5, 6]]

IMPORTANT: Return ONLY the array, with no additional text, quotes, or formatting.
"""


def build_owner_prompt(training_pairs: List[Dict],
                       test_input: List[List[int]]) -> str:
    """Render the current ARC Prize baseline prompt byte-for-byte in JSON form."""
    training_examples = ''
    for index, pair in enumerate(training_pairs):
        training_examples += f'--Example {index}-- \n\n INPUT: \n\n'
        training_examples += json.dumps(pair['input']) + '\n\n'
        training_examples += 'OUTPUT: \n\n'
        training_examples += json.dumps(pair['output']) + '\n\n'
    return OWNER_PROMPT_TEMPLATE.format(
        training_examples=training_examples,
        test_input=json.dumps(test_input),
    )


class ARCPrizeGenInferencer(GenInferencer):
    """Run the ARC Prize extraction retry with the evaluated model.

    The official benchmarking harness asks the same model to extract a JSON
    grid only when the primary response cannot be parsed. Keeping this retry
    inside inference makes the behavior work for both local and API models and
    avoids depending on a separately configured endpoint during evaluation.
    """

    def __init__(self,
                 *args,
                 enable_second_pass: bool = True,
                 second_pass_max_out_len: int = 4096,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enable_second_pass = enable_second_pass
        self.second_pass_max_out_len = second_pass_max_out_len

    @staticmethod
    def _prediction_text(prediction) -> Optional[str]:
        if isinstance(prediction, str):
            return prediction
        if isinstance(prediction, dict):
            value = prediction.get('prediction')
            return value if isinstance(value, str) else None
        return None

    @staticmethod
    def _replace_prediction_text(prediction, text):
        if not isinstance(prediction, dict):
            return text
        updated = prediction.copy()
        updated['prediction'] = text
        return updated

    def _generate_with_second_pass(self, generate, templates, max_out_len,
                                   **kwargs):
        predictions = list(
            generate(templates, max_out_len=max_out_len, **kwargs))
        if not self.enable_second_pass:
            return predictions

        failed_indexes = []
        extraction_prompts = []
        for index, prediction in enumerate(predictions):
            prediction_text = self._prediction_text(prediction)
            if prediction_text is None or extract_solution(
                    prediction_text) is not None:
                continue
            failed_indexes.append(index)
            extraction_prompts.append(
                PromptList([
                    dict(
                        role='HUMAN',
                        prompt=SECOND_PASS_EXTRACTION_PROMPT.format(
                            response=prediction_text),
                    )
                ]))

        if not extraction_prompts:
            return predictions

        extracted_responses = generate(
            extraction_prompts,
            max_out_len=self.second_pass_max_out_len,
            **kwargs,
        )
        for index, response in zip(failed_indexes, extracted_responses):
            response_text = self._prediction_text(response)
            parsed = _parse_second_pass_response(response_text)
            if parsed is None:
                continue
            predictions[index] = self._replace_prediction_text(
                predictions[index], json.dumps(parsed))
        return predictions

    def inference(self, *args, **kwargs):
        generate = self.model.generate_from_template

        def generate_with_second_pass(templates, max_out_len, **gen_kwargs):
            return self._generate_with_second_pass(
                generate,
                templates,
                max_out_len,
                **gen_kwargs,
            )

        self.model.generate_from_template = generate_with_second_pass
        try:
            return super().inference(*args, **kwargs)
        finally:
            self.model.generate_from_template = generate


@LOAD_DATASET.register_module()
class ARCPrizeDataset(BaseDataset):
    task_file_names = {
        'arc_agi_1': [
            '2072aba6.json', 'bb52a14b.json', '136b0064.json', 'ea9794b1.json',
            '40f6cd08.json', 'f5aa3634.json', '7039b2d7.json', '712bf12e.json',
            '9b365c51.json', 'ccd554ac.json', 'f9d67f8b.json', '03560426.json',
            'e2092e0c.json', '8fbca751.json', '42918530.json', 'c64f1187.json',
            '00576224.json', '705a3229.json', 'af24b4cc.json', '81c0276b.json',
            'f21745ec.json', '8dae5dfc.json', '4e469f39.json', '695367ec.json',
            'dc2aa30b.json', 'b9630600.json', '770cc55f.json', '3391f8c0.json',
            'c1990cce.json', '1da012fc.json', '50a16a69.json', '212895b5.json',
            'e69241bd.json', '692cd3b6.json', '0bb8deee.json', '9772c176.json',
            '22a4bbc2.json', 'ca8de6ea.json', 'dc2e9a9d.json', '4aab4007.json',
            'cfb2ce5a.json', '9f27f097.json', '2c737e39.json', '84db8fc4.json',
            'e1baa8a4.json', 'ea959feb.json', '4f537728.json', '47996f11.json',
            'bf32578f.json', 'aee291af.json', '5d2a5c43.json', '2546ccf6.json',
            'e57337a4.json', 'd4b1c2b1.json', '20981f0e.json', '05a7bcf2.json',
            'fc754716.json', '6ad5bdfd.json', 'e88171ec.json', '1acc24af.json',
            '34b99a2b.json', 'e78887d1.json', '4acc7107.json', '137f0df0.json',
            '62b74c02.json', '50aad11f.json', '642d658d.json', '64a7c07e.json',
            'bd14c3bf.json', '73c3b0d8.json', 'e0fb7511.json', 'c7d4e6ad.json',
            '85b81ff1.json', 'e760a62e.json', 'ca8f78db.json', 'd931c21c.json',
            'aab50785.json', 'ac605cbb.json', '3194b014.json', '68b67ca3.json',
            'e7b06bea.json', 'e5790162.json', 'da2b0fe3.json', '0becf7df.json',
            'fe9372f3.json', 'd56f2372.json', 'e66aafb8.json', 'b7999b51.json',
            '2697da3f.json', '516b51b7.json', '9a4bb226.json', '195ba7dc.json',
            '310f3251.json', '639f5a19.json', '0d87d2a6.json', 'c663677b.json',
            'e74e1818.json', '69889d6e.json', 'f45f5ca7.json', '8597cfd7.json',
            '0c9aba6e.json', 'e9b4f6fc.json', 'e7639916.json', '5207a7b5.json',
            'e4075551.json', '90347967.json', '9ddd00f0.json', '4b6b68e5.json',
            'e9c9d9a1.json', '2f0c5170.json', '58e15b12.json', 'd37a1ef5.json',
            '62ab2642.json', 'b457fec5.json', 'c97c0139.json', 'ac0c5833.json',
            '7d419a02.json', '4ff4c9da.json', '4cd1b7b2.json', '27a77e38.json',
            '66f2d22f.json', '2a5f8217.json', 'c074846d.json', 'c6e1b8da.json',
            '319f2597.json', '94be5b80.json', '55783887.json', '60c09cac.json',
            'f823c43c.json', 'd492a647.json', 'e681b708.json', '15663ba9.json',
            'a3f84088.json', '103eff5b.json', '5a5a2103.json', '1e97544e.json',
            '009d5c81.json', 'ed74f2f2.json', 'ce039d91.json', 'baf41dbf.json',
            '3490cc26.json', 'ce8d95cc.json', '3f23242b.json', '1d0a4b61.json',
            '8719f442.json', 'd94c3b52.json', '4c177718.json', '59341089.json',
            '3ee1011a.json', 'f5c89df1.json', '5833af48.json', 'd4c90558.json',
            '88207623.json', '833dafe3.json', '070dd51e.json', '3ed85e70.json',
            '21f83797.json', '7c8af763.json', '5783df64.json', 'a57f2f04.json',
            'e9ac8c9e.json', 'aa18de87.json', '505fff84.json', '5ffb2104.json',
            '42a15761.json', '1a2e2828.json', '0607ce86.json', '84f2aca1.json',
            '456873bc.json', '903d1b4a.json', '0f63c0b9.json', '54db823b.json',
            'ad7e01d0.json', '8e2edd66.json', '79fb03f4.json', '4364c1c4.json',
            'e7a25a18.json', 'e133d23d.json', 'e21a174a.json', '55059096.json',
            'e95e3d8e.json', '94414823.json', '9356391f.json', '15113be4.json',
            'ba9d41b8.json', '52fd389e.json', 'de493100.json', '9c56f360.json',
            'c92b942c.json', '97239e3d.json', 'b0f4d537.json', '19bb5feb.json',
            '506d28a5.json', '5b692c0f.json', 'ef26cbf6.json', 'e345f17b.json',
            '7d1f7ee8.json', 'ac3e2b04.json', '551d5bf1.json', 'fb791726.json',
            '2037f2c7.json', 'e6de6e8f.json', '3d31c5b3.json', 'd19f7514.json',
            '1d398264.json', '358ba94e.json', '696d4842.json', '08573cc6.json',
            '7e02026e.json', '7953d61e.json', 'c3202e5a.json', '351d6448.json',
            'fea12743.json', '12422b43.json', 'b942fd60.json', 'bcb3040b.json',
            'e41c6fd3.json', 'a59b95c0.json', '3a301edc.json', '0b17323b.json',
            'da515329.json', '96a8c0cd.json', '6f473927.json', '9def23fe.json',
            'c35c1b4c.json', 'be03b35f.json', '604001fa.json', 'd304284e.json',
            'cb227835.json', 'e9bb6954.json', 'ac2e8ecf.json', '1e81d6f9.json',
            '72207abc.json', '37d3e8b2.json', 'c8b7cc0f.json', 'a096bf4d.json',
            '1c02dbbe.json', 'fd096ab6.json', '9bebae7a.json', '25094a63.json',
            'b7fb29bc.json', 'aa4ec2a5.json', '50f325b5.json', '423a55dc.json',
            'b0722778.json', 'e7dd8335.json', 'f3cdc58f.json', 'cad67732.json',
            '256b0a75.json', 'd282b262.json', '58743b76.json', '6df30ad6.json',
            '9110e3c5.json', '48f8583b.json', 'a680ac02.json', '642248e4.json',
            '2685904e.json', '48131b3c.json', 'b7cb93ac.json', '73182012.json',
            'df8cc377.json', '3b4c2228.json', '93c31fbe.json', '8ee62060.json',
            '9b2a60aa.json', 'f0df5ff0.json', '917bccba.json', 'ed98d772.json',
            'bf89d739.json', 'f3e62deb.json', '11e1fe23.json', 'bbb1b8b6.json',
            'f4081712.json', '817e6c09.json', '45bbe264.json', 'f3b10344.json',
            'fafd9572.json', 'b7f8a4d8.json', '2c0b0aff.json', '8cb8642d.json',
            '67c52801.json', 'd47aa2ff.json', '0934a4d8.json', '60a26a3e.json',
            'cf133acc.json', '5289ad53.json', '16b78196.json', '09c534e7.json',
            'f83cb3f6.json', 'd017b73f.json', 'b20f7c8b.json', '5af49b42.json',
            '18419cfa.json', '929ab4e9.json', '6a11f6da.json', '17cae0c1.json',
            'e99362f0.json', '1c56ad9f.json', '8a371977.json', 'e633a9e5.json',
            'c658a4bd.json', 'bc4146bd.json', '67636eac.json', '4e45f183.json',
            '17b80ad2.json', '94133066.json', 'e1d2900e.json', 'a934301b.json',
            '0a2355a6.json', '45737921.json', '332efdb3.json', '7bb29440.json',
            'f9a67cb5.json', 'a8610ef7.json', '32e9702f.json', '0c786b71.json',
            '626c0bcc.json', 'aa300dc3.json', 'c62e2108.json', '0692e18c.json',
            'af22c60d.json', '992798f6.json', 'c48954c1.json', '5b526a93.json',
            'ae58858e.json', 'ff72ca3e.json', '2b01abd0.json', '7d18a6fb.json',
            '963f59bc.json', '759f3fd3.json', '7c9b52a0.json', '4852f2fa.json',
            '14754a24.json', 'c87289bb.json', '845d6e51.json', '281123b4.json',
            '79369cc6.json', '0a1d4ef5.json', '477d2879.json', '72a961c9.json',
            '67b4a34d.json', 'e5c44e8f.json', 'bf699163.json', '13713586.json',
            '27f8ce4f.json', '95a58926.json', '15696249.json', 'd2acf2cb.json',
            '140c817e.json', '1990f7a8.json', '782b5218.json', '8b28cd80.json',
            '92e50de0.json', 'e619ca6e.json', '5b6cbef5.json', '575b1a71.json',
            '66e6c45b.json', '31adaf00.json', '6ea4a07e.json', 'f0afb749.json',
            '00dbd492.json', 'b1fc8b8e.json', 'fd4b2b02.json', 'b15fca0b.json',
            'a04b2602.json', '20818e16.json', '762cd429.json', '29700607.json',
            'd5c634a2.json', 'a406ac07.json', '8ba14f53.json', '184a9768.json',
            '12997ef3.json', 'dd2401ed.json', 'f8be4b64.json', '12eac192.json',
            '31d5ba1a.json', 'b4a43f3b.json', '7ee1c6ea.json', '9b4c17c4.json',
            '981571dc.json', '93b4f4b3.json', '9caba7c3.json', '891232d6.json',
            '85fa5666.json', '0e671a1a.json', '73ccf9c2.json', '414297c0.json',
            'e872b94a.json', '99306f82.json', '3979b1a8.json', '2753e76c.json',
            '1c0d0a4b.json', '292dd178.json', 'cd3c21df.json', '33b52de3.json',
            'ecaa0ec1.json', '896d5239.json', '1a6449f1.json', '9c1e755f.json'
        ],
        'arc_agi_2': [
            '136b0064.json', '898e7135.json', 'a32d8b75.json', '7491f3cf.json',
            '2d0172a1.json', '71e489b6.json', '21897d95.json', '446ef5d2.json',
            '8e5c0c38.json', '3e6067c3.json', '7ed72f31.json', '1ae2feb7.json',
            '9aaea919.json', '8b9c3697.json', '53fb4810.json', 'e8686506.json',
            'a47bf94d.json', '4a21e3da.json', '8f3a5a89.json', 'edb79dae.json',
            'fc7cae8d.json', 'b0039139.json', 'db695cfb.json', '409aa875.json',
            '3a25b0d8.json', '2c181942.json', 'b6f77b65.json', '13e47133.json',
            '64efde09.json', '89565ca0.json', 'd59b0160.json', '135a2760.json',
            'cbebaa4b.json', '5545f144.json', 'bf45cf4b.json', '20a9e565.json',
            '221dfab4.json', 'a251c730.json', 'faa9f03d.json', '8b7bacbf.json',
            'e376de54.json', '8698868d.json', 'b99e7126.json', '4c7dc4dd.json',
            'd8e07eb2.json', '291dc1e1.json', '62593bfd.json', '7b5033c1.json',
            '4c3d4a41.json', '4e34c42c.json', '16de56c4.json', '6ffbe589.json',
            '88bcf3b4.json', 'a25697e4.json', '9bbf930d.json', '195c6913.json',
            '332f06d7.json', 'dd6b8c4b.json', '31f7f899.json', '7b0280bc.json',
            '5dbc8537.json', 'dfadab01.json', '88e364bc.json', '9385bd28.json',
            '20270e3b.json', '247ef758.json', '269e22fb.json', '1818057f.json',
            '271d71e2.json', 'da515329.json', '3dc255db.json', '142ca369.json',
            'eee78d87.json', 'aa4ec2a5.json', 'db0c5428.json', 'f931b4a8.json',
            'e3721c99.json', '45a5af55.json', '35ab12c3.json', '2b83f449.json',
            'cb2d8a2c.json', 'a6f40cea.json', 'abc82100.json', 'a395ee82.json',
            'b5ca7ac4.json', '67e490f4.json', '7c66cb00.json', '0934a4d8.json',
            '16b78196.json', 'b9e38dc0.json', '58490d8a.json', '4c416de3.json',
            '7666fa5d.json', '7b80bb43.json', 'de809cff.json', 'e12f9a14.json',
            '6e4f6532.json', '6e453dd6.json', '78332cb0.json', '8f215267.json',
            '58f5dbd5.json', '581f7754.json', '800d221b.json', 'd35bdbdc.json',
            'b10624e5.json', 'dbff022c.json', '65b59efc.json', '97d7923e.json',
            'c7f57c3e.json', '2ba387bc.json', '28a6681f.json', '5961cc34.json',
            '80a900e0.json', '981571dc.json', 'f560132c.json', 'e87109e9.json',
            '38007db0.json', '36a08778.json', 'c4d067a0.json', '7b3084d4.json'
        ],
    }

    @staticmethod
    def load(
        path: str,
        version: str,
        task_ids: Optional[List[str]] = None,
        protocol: str = 'legacy',
    ):
        if protocol not in ('legacy', 'owner'):
            raise ValueError('protocol must be either "legacy" or "owner"')
        task_file_dir = get_data_path(path)

        dataset = []

        task_file_name_list = os.listdir(task_file_dir)
        if protocol == 'owner':
            task_file_name_list = sorted(task_file_name_list)
        for task_file_name in task_file_name_list:
            if task_file_name not in ARCPrizeDataset.task_file_names[version]:
                continue
            task_id = os.path.splitext(task_file_name)[0]
            if task_ids is not None and task_id not in task_ids:
                continue
            with open(os.path.join(task_file_dir, task_file_name),
                      'r') as file:
                task = json.load(file)
                if protocol == 'legacy':
                    test_pair = task['test'][0]
                    dataset.append({
                        'training_data': task['train'],
                        'input_test_data': test_pair['input'],
                        'output_test_data': test_pair['output'],
                    })
                    continue

                # Owner protocol keeps every test pair as an inference row and
                # retains task identity for task-level pair aggregation.
                for pair_index, test_pair in enumerate(task['test']):
                    reference = {
                        'task_id': task_id,
                        'pair_index': pair_index,
                        'num_pairs': len(task['test']),
                        'output': test_pair['output'],
                    }
                    dataset.append({
                        'task_id':
                        task_id,
                        'pair_index':
                        pair_index,
                        'training_data':
                        task['train'],
                        'input_test_data':
                        test_pair['input'],
                        'output_test_data':
                        reference,
                        'prompt':
                        build_owner_prompt(task['train'], test_pair['input']),
                    })

        return Dataset.from_list(dataset)


class ARCPrizeEvaluator(BaseEvaluator):

    def __init__(
        self,
        protocol: str = 'legacy',
        attempt_aggregation: str = 'mean',
        attempts_per_pair: Optional[int] = None,
    ) -> None:
        if protocol not in ('legacy', 'owner'):
            raise ValueError('protocol must be either "legacy" or "owner"')
        if attempt_aggregation not in ('mean', 'any'):
            raise ValueError(
                'attempt_aggregation must be either "mean" or "any"')
        if attempts_per_pair is not None and attempts_per_pair < 1:
            raise ValueError('attempts_per_pair must be positive')
        if protocol == 'legacy' and (attempt_aggregation != 'mean'
                                     or attempts_per_pair is not None):
            raise ValueError('attempt aggregation requires owner protocol')
        self.protocol = protocol
        self.attempt_aggregation = attempt_aggregation
        self.attempts_per_pair = attempts_per_pair

    def evaluate(
        self,
        k: int,
        n: int,
        original_dataset: Dataset,
        **score_kwargs,
    ) -> Dict:
        """Evaluate replicas as independent runs or owner-style attempts.

        OpenCompass normally evaluates each of the ``n`` dataset replicas
        independently and averages their scores. ARC-AGI-2 instead treats the
        replicas as attempts for the same test pair: the pair is correct when
        any attempt exactly matches, then pair scores are averaged within each
        task. Keep the default BaseEvaluator behavior for ARC-AGI-1 and legacy
        configs, and opt into owner attempt aggregation explicitly.
        """
        if self.protocol == 'legacy' or self.attempt_aggregation != 'any':
            return super().evaluate(k, n, original_dataset, **score_kwargs)

        if self.attempts_per_pair is not None and n != self.attempts_per_pair:
            raise ValueError(
                f'expected {self.attempts_per_pair} attempts per pair, got {n}'
            )
        if n < 1 or len(original_dataset) % n != 0:
            raise ValueError(
                'replicated ARC dataset size must be divisible by n')

        predictions = score_kwargs.get('predictions')
        references = score_kwargs.get('references')
        if predictions is None or references is None:
            raise ValueError(
                'owner attempt aggregation requires predictions and references'
            )
        if len(predictions) != len(references):
            raise ValueError(
                'Predictions and references must have the same length')
        if len(predictions) != len(original_dataset):
            raise ValueError(
                'Predictions must cover every replicated dataset row')

        real_size = len(original_dataset) // n
        for replica_index in range(1, n):
            start = replica_index * real_size
            for row_index in range(real_size):
                if references[start + row_index] != references[row_index]:
                    raise ValueError(
                        'ARC attempt replicas must preserve reference order')

        processed_predictions = self.pred_postprocess(predictions)
        attempt_result = self.score(
            predictions=processed_predictions,
            references=references,
        )
        attempt_details = attempt_result.pop('details')
        if len(attempt_details) != n * real_size:
            raise ValueError('ARC attempt details do not match dataset size')

        task_pair_scores = defaultdict(list)
        pair_details = []
        for row_index in range(real_size):
            attempts = [
                attempt_details[replica_index * real_size + row_index]
                for replica_index in range(n)
            ]
            task_id = attempts[0]['task_id']
            pair_index = attempts[0]['pair_index']
            if any(detail['task_id'] != task_id
                   or detail['pair_index'] != pair_index
                   for detail in attempts):
                raise ValueError(
                    'ARC attempt replicas must preserve task and pair identity'
                )

            pair_correct = int(any(detail['correct'] for detail in attempts))
            task_pair_scores[task_id].append(pair_correct)
            pair_details.append({
                'task_id':
                task_id,
                'pair_index':
                pair_index,
                'correct':
                pair_correct,
                'correct_percentage':
                max(detail['correct_percentage'] for detail in attempts),
                'attempt_count':
                n,
                'attempt_correct': [detail['correct'] for detail in attempts],
                'attempt_parse_success':
                [detail['parse_success'] for detail in attempts],
                'attempt_empty_list_prediction':
                [detail['empty_list_prediction'] for detail in attempts],
                'generated_solution':
                [detail['generated_solution'] for detail in attempts],
                'raw_prediction':
                [detail['raw_prediction'] for detail in attempts],
            })

        task_scores = [np.mean(scores) for scores in task_pair_scores.values()]
        pair_scores = [detail['correct'] for detail in pair_details]
        return {
            'accuracy': float(np.mean(task_scores)) if task_scores else 0.0,
            'pair_accuracy':
            (float(np.mean(pair_scores)) if pair_scores else 0.0),
            'task_count': len(task_scores),
            'pair_count': len(pair_scores),
            'attempt_count': len(attempt_details),
            'attempts_per_pair': n,
            'parse_success_rate': attempt_result['parse_success_rate'],
            'details': pair_details,
        }

    def score(self, predictions: List[str], references: List[Any]) -> Dict:
        if self.protocol == 'legacy':
            return self._score_legacy(predictions, references)

        parsed_predictions = [extract_solution(text) for text in predictions]

        task_pair_scores = defaultdict(list)
        details = []
        for index, (raw_prediction,
                    reference) in enumerate(zip(predictions, references)):
            prediction = parsed_predictions[index]
            if isinstance(reference, dict):
                task_id = reference['task_id']
                pair_index = reference['pair_index']
                expected = reference['output']
            else:
                # Backward compatibility for archived configs/results.
                task_id = str(len(details))
                pair_index = 0
                expected = reference

            is_correct = prediction == expected
            if _is_grid(prediction):
                _, correct_percentage = compare_solutions_with_padding(
                    prediction, expected, pad_value=-1)
            else:
                correct_percentage = 0.0
            task_pair_scores[task_id].append(1 if is_correct else 0)
            details.append({
                'task_id': task_id,
                'pair_index': pair_index,
                'correct': 1 if is_correct else 0,
                'correct_percentage': correct_percentage,
                'parse_success': prediction is not None,
                'empty_list_prediction': prediction == [],
                'generated_solution': prediction,
                'raw_prediction': raw_prediction,
            })

        task_scores = [np.mean(scores) for scores in task_pair_scores.values()]
        pair_scores = [detail['correct'] for detail in details]
        return {
            'accuracy':
            float(np.mean(task_scores)) if task_scores else 0.0,
            'pair_accuracy':
            (float(np.mean(pair_scores)) if pair_scores else 0.0),
            'task_count':
            len(task_scores),
            'pair_count':
            len(pair_scores),
            'parse_success_rate':
            float(np.mean([detail['parse_success']
                           for detail in details])) if details else 0.0,
            'details':
            details,
        }

    @staticmethod
    def _score_legacy(
        predictions: List[str],
        references: List[List[int]],
    ) -> Dict:
        """Preserve the pre-MR parser, reference schema, and output fields."""
        accuracy = []
        details = []
        for prediction, reference in zip(
                map(_extract_solution_legacy, predictions),
                references,
        ):
            is_correct, correct_percentage = compare_solutions_with_padding(
                prediction,
                reference,
                pad_value=-1,
            )
            details.append({
                'correct': 1 if is_correct else 0,
                'correct_percentage': correct_percentage,
                'generated_solution': prediction,
            })
            accuracy.append(1 if is_correct else 0)
        return {'accuracy': np.mean(accuracy), 'details': details}


def _is_grid(value) -> bool:
    return (isinstance(value, list) and len(value) > 0
            and all(isinstance(row, list) and len(row) > 0 for row in value)
            and len({len(row)
                     for row in value}) == 1 and all(
                         type(cell) is int and 0 <= cell <= 9 for row in value
                         for cell in row))


def _is_owner_list_of_lists(value, require_non_empty: bool = False) -> bool:
    """Use the structure checks in the ARC Prize baseline parser."""
    return (isinstance(value, list) and (not require_non_empty or bool(value))
            and all(isinstance(row, list) for row in value))


def _extract_boxed_grid(text: str) -> Optional[List[List[int]]]:
    """Match the owner's first-pass ``\\boxed{...}`` response parser."""
    match = re.search(r'\\boxed\{(.*?)\}', text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None
    return value if _is_owner_list_of_lists(value) else None


def _backscan_json_grid(text: str) -> Optional[List[List[int]]]:
    """Backscan the response for its final complete JSON value."""
    last_bracket_index = -1
    closing_bracket = None
    for index in range(len(text) - 1, -1, -1):
        if text[index] in (']', '}'):
            last_bracket_index = index
            closing_bracket = text[index]
            break
    if last_bracket_index == -1:
        return None

    opening_bracket = '[' if closing_bracket == ']' else '{'
    bracket_counter = 1
    start_index = -1
    for index in range(last_bracket_index - 1, -1, -1):
        if text[index] == closing_bracket:
            bracket_counter += 1
        elif text[index] == opening_bracket:
            bracket_counter -= 1
            if bracket_counter == 0:
                start_index = index
                break
    if start_index == -1:
        return None

    try:
        value = json.loads(text[start_index:last_bracket_index + 1])
    except json.JSONDecodeError:
        return None
    return value if _is_owner_list_of_lists(value,
                                            require_non_empty=True) else None


def _extract_solution_legacy(text: str) -> List[List[int]]:
    """Preserve the parser used by archived ARC configs before this MR."""
    try:
        start = text.index('[[')
        end = text.index(']]', start) + 2
        value = ast.literal_eval(text[start:end])
        if (all(isinstance(row, list) for row in value) and all(
                all(isinstance(cell, int) for cell in row) for row in value)):
            return value
    except (ValueError, SyntaxError):
        pass
    return [[0]]


def extract_solution(text: str) -> Optional[List[List[int]]]:
    """Apply the current ARC Prize baseline response-parser order."""
    if not isinstance(text, str):
        return None
    boxed_grid = _extract_boxed_grid(text)
    if boxed_grid is not None:
        return boxed_grid
    return _backscan_json_grid(text)


def _parse_second_pass_response(
        assistant_content: str) -> Optional[List[List[int]]]:
    """Mirror the owner's OpenAI adapter postprocessing after call two."""
    if not isinstance(assistant_content, str):
        return None

    if '```' in assistant_content:
        for block in assistant_content.split('```'):
            if block.strip() and not block.strip().startswith('json'):
                assistant_content = block.strip()
                break
    assistant_content = assistant_content.strip()

    if assistant_content and not assistant_content.startswith('['):
        start_index = assistant_content.find('[[')
        if start_index >= 0:
            end_index = assistant_content.rfind(']]') + 2
            if end_index > start_index:
                assistant_content = assistant_content[start_index:end_index]

    try:
        value = json.loads(assistant_content)
    except json.JSONDecodeError:
        pattern = (r'\[\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]'
                   r'(?:\s*,\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\])*\s*\]')
        match = re.search(pattern, assistant_content)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if isinstance(value, list) and _is_owner_list_of_lists(value):
        return value
    if isinstance(value, dict) and 'response' in value:
        return value.get('response')
    return None


def pad_array_with_value(array, target_shape, pad_value):
    padded_array = np.full(target_shape, pad_value, dtype=int)
    for i in range(len(array)):
        padded_array[i, :len(array[i])] = array[i]
    return padded_array


def compare_solutions_with_padding(generated_output: List[int],
                                   correct_output: List[int],
                                   pad_value=-1):
    if not generated_output or not correct_output:
        return False, 0.0
    max_rows = max(len(generated_output), len(correct_output))
    max_cols = max(max(map(len, generated_output)),
                   max(map(len, correct_output)))
    target_shape = (max_rows, max_cols)

    padded_generated = pad_array_with_value(generated_output, target_shape,
                                            pad_value)
    padded_correct = pad_array_with_value(correct_output, target_shape,
                                          pad_value)

    total_pixels = max_rows * max_cols
    correct_pixels = np.sum((padded_generated == padded_correct)
                            & (padded_generated != pad_value)
                            & (padded_correct != pad_value))
    correct_percentage = (correct_pixels / total_pixels) * 100

    is_correct = (correct_pixels == total_pixels)

    return is_correct, correct_percentage
