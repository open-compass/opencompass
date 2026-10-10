"""EarthSE datasets and evidence-backed evaluators.

The public dataset artifacts are revision-pinned in the accompanying
configuration.  The implementation is adapted from frozen evidence E1.  Its
manifest establishes these source-relative paths and content hashes, but does
not establish an upstream repository or revision:

* ``evaluation/Earth_Iron_Silver.py`` sha256
  ``b023980e1fe4e6bf843d118afe04f149f90628f4b59855478c8548e710d2ca24``
* ``evaluation/Earth_Gold.py`` sha256
  ``6f85010f7bfdf4f68dbcf59be884b5c1bfda4903e43c2944089efb59763d8ca8``
* ``evaluation/prompts.py`` sha256
  ``cf1236f0e45f0813bd5e976d8727cf739b1bfb7bba0029049e47bddae68f9172``
* ``evaluation/utils.py`` sha256
  ``720e06ea69674ffc9aa58ad6d35011a5516d9757ceaaa1a190dda64e307ffd3f``
* ``evaluation/show_results.py`` sha256
  ``f05a843c8bb8353b6d721de5b0a6533eb5ffc7f0145f122061a3b3ae43d7da43``

The frozen E1 snapshot also contains no license file or license metadata.
Upstream origin, revision, and reuse terms therefore remain unknown and
require maintainer or legal review before merging or redistribution.
"""

import copy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import BoundedSemaphore
from typing import Dict, Iterable, List, Optional

from datasets import Dataset, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.registry import (ICL_EVALUATORS, ICL_INFERENCERS,
                                  LOAD_DATASET, TEXT_POSTPROCESSORS)
from opencompass.utils import build_model_from_cfg

from .base import BaseDataset


SYSTEM_PROMPT = 'You are a helpful assistant.'
GOLD_PROMPT = 'Please respond to the following question in 80 words or less.\n'
REASONING_SEPARATOR = '<|earthse_reasoning_end|>'

FREE_FORM_JUDGE_PROMPT = """
Please act as an expert evaluator and determine which of the following two answers is better.

**Evaluation Criteria:**
1. Assess how well each answer addresses the original question. Closer alignment is better.
2. Evaluate the scientific accuracy and logical coherence of each answer. More rigorous and professional reasoning is preferred.
3. Consider the relevance and depth of detail. More relevant and well-supported details indicate a better answer.
4. It is not the case that the longer the answer, the better. If the answer is long but does not meet the above requirements, it is not a good answer.

**Instructions:**
1. Do **not** generate a new answer to the original question. Your task is only to evaluate the two provided answers.
2. Based on the criteria above, choose which answer is better.
3. Your response must be **only** one letter: `A` or `B`.
4. Do **not** provide explanations, commentary, or corrections, even if there are errors in the inputs.
5. This is purely an evaluation task.

**[Question Start]**
<QUES>
**[Question End]**

**[Answer A Start]**
<ANS_A>
**[Answer A Start]**

**[Answer B Start]**
<ANS_B>
**[Answer B Start]**

**The better answer is:**
"""

GOLD_JUDGE_PROMPT = """
Please rank the reference dialogue among all dialogues based on the following criteria:

**Evaluation Criteria:**
1. Depth of Reflection: A dialogue is considered high-quality if it contains in-depth analysis and reflection on the topic.
2. Novelty of Approach: A dialogue is considered high-quality if it proposes innovative solutions or unique insights.

You only need to output **an integer** representing the ranking of the reference dialogue among all dialogues (1 being the best, higher numbers indicating lower rankings).

**[Reference Dialogue Start]**
<Dialogue_1>
**[Reference Dialogue End]**

**[Other Dialogues Start]**
<Dialogue_2>
**[Other Dialogues Start]**

**The ranking of the reference dialogue among all dialogues (An integer between 1 and <NUM>):**
"""


def earthse_prompt(question: str, question_type: str) -> str:
    """Render the prompt used by the pinned EarthSE evaluation."""
    templates = {
        'multiple_choice': '''
Please respond to the following multiple-choice question by providing your answer as a single letter, without any additional text.

{question}

The answer is (single letter):
''',
        'true_false': '''
Please answer the following true or false question with "True" or "False" without adding any additional text.

{question}

The answer is ("True" or "False"):
''',
        'fill_in_the_blank': '''
Please answer the fill-in-the-blank question below with lowercase words or phrases. If your answer contains multiple words or phrases, please separate them with commas. No additional text is required.

{question}

The answer is:
''',
        'free_form': '''
Please answer the following question:

{question}

The answer is:
''',
    }
    try:
        return templates[question_type].format(question=question)
    except KeyError as exc:
        raise ValueError(
            f'Unsupported EarthSE question type: {question_type}') from exc


@TEXT_POSTPROCESSORS.register_module()
def earthse_extract_content(value: str,
                            separator: str = REASONING_SEPARATOR) -> str:
    """Prefer canonical content after a configured reasoning separator."""
    if separator and separator in value:
        return value.rsplit(separator, 1)[1]
    return value


def _load_source(path: str, split: str, revision: Optional[str] = None):
    """Load either a materialized Parquet file or a public HF dataset."""
    if path.endswith('.parquet'):
        if not Path(path).is_file():
            raise FileNotFoundError(f'EarthSE artifact does not exist: {path}')
        return load_dataset('parquet', data_files={split: path}, split=split)
    kwargs = {'split': split}
    if revision:
        kwargs['revision'] = revision
    return load_dataset(path, **kwargs)


@LOAD_DATASET.register_module()
class EarthSEDataset(BaseDataset):
    """Load any combination of the four Earth-Iron/Silver QA splits.

    ``paths`` maps an official split name to a local Parquet path.  It is used
    by reproducible/offline runs.  The public configuration instead supplies
    a pinned Hugging Face repository through ``path`` and ``revision``.
    """

    SPLITS = ('multiple_choice', 'true_false', 'fill_in_the_blank',
              'free_form')

    @staticmethod
    def load(path: Optional[str] = None,
             paths: Optional[Dict[str, str]] = None,
             dataset_name: str = 'Earth-Silver',
             question_types: Optional[Iterable[str]] = None,
             revision: Optional[str] = None,
             max_samples_per_split: Optional[int] = None,
             prompt_mode: str = 'zero-shot',
             **kwargs):
        if prompt_mode != 'zero-shot':
            raise NotImplementedError('EarthSE defines only zero-shot prompts')
        selected = list(question_types or EarthSEDataset.SPLITS)
        unsupported = [x for x in selected if x not in EarthSEDataset.SPLITS]
        if unsupported:
            raise ValueError(f'Unsupported EarthSE splits: {unsupported}')
        if paths is None and path is None:
            raise ValueError('EarthSEDataset requires path or paths')
        if max_samples_per_split is not None and max_samples_per_split <= 0:
            raise ValueError('max_samples_per_split must be positive')
        if paths is not None:
            missing_sources = [name for name in selected if not paths.get(name)]
            if missing_sources:
                raise FileNotFoundError(
                    f'No artifact configured for {dataset_name}/'
                    f'{missing_sources[0]}')

        rows = []
        prefix = dataset_name.lower().replace('-', '_')
        for question_type in selected:
            source = paths.get(question_type) if paths else path
            if not source:
                raise FileNotFoundError(
                    f'No artifact configured for {dataset_name}/'
                    f'{question_type}')
            split = _load_source(source, question_type,
                                 None if paths else revision)
            required = {
                'idx', 'question', 'reasoning_chain', 'answer', 'task',
                'sphere', 'subject', 'sub_discipline'
            }
            missing = required.difference(split.column_names)
            if missing:
                raise ValueError(
                    f'{dataset_name}/{question_type} is missing columns: '
                    f'{sorted(missing)}')
            for index, item in enumerate(split):
                if (max_samples_per_split is not None
                        and index >= max_samples_per_split):
                    break
                row = dict(item)
                source_idx = str(row['idx'])
                row.update(
                    dataset_name=dataset_name,
                    question_type=question_type,
                    source_idx=source_idx,
                    sample_id=f'{prefix}_{question_type}:{source_idx}',
                    prompt=earthse_prompt(row['question'], question_type),
                )
                rows.append(row)
        ids = [row['sample_id'] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError('EarthSE source IDs are not unique')
        return Dataset.from_list(rows)


@LOAD_DATASET.register_module()
class EarthSECombinedDataset(BaseDataset):
    """Load several QA tiers into one native inference/evaluation stream.

    EarthSE's Table 4 applies the same prompt and evaluator to Earth-Silver
    and Earth-Iron.  Keeping both tiers in one dataset lets the framework's
    request pools consume the full experiment without a tier boundary.
    """

    @staticmethod
    def load(sources: List[dict],
             max_samples_per_split: Optional[int] = None,
             **kwargs):
        if not sources:
            raise ValueError('EarthSECombinedDataset requires sources')
        rows = []
        for source in sources:
            source_cfg = copy.deepcopy(dict(source))
            source_cfg['max_samples_per_split'] = max_samples_per_split
            rows.extend(dict(row) for row in EarthSEDataset.load(**source_cfg))
        ids = [row['sample_id'] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError('Combined EarthSE source IDs are not unique')
        return Dataset.from_list(rows)


@LOAD_DATASET.register_module()
class Earth_Silver_MCQDataset(EarthSEDataset):
    """Backward-compatible name for the historical Silver MCQ config."""

    @staticmethod
    def load(path: str,
             prompt_mode: str = 'zero-shot',
             revision: Optional[str] = None,
             **kwargs):
        return EarthSEDataset.load(
            path=path,
            dataset_name='Earth-Silver',
            question_types=['multiple_choice'],
            revision=revision,
            prompt_mode=prompt_mode,
        )


@LOAD_DATASET.register_module()
class EarthSEGoldDataset(BaseDataset):
    """Load Earth-Gold and expand each dialogue into the official 3 trials."""

    @staticmethod
    def load(path: str,
             revision: Optional[str] = None,
             repetitions: int = 3,
             **kwargs):
        if repetitions != 3:
            raise ValueError('The Earth-Gold protocol fixes repetitions=3')
        split = _load_source(path, 'train', revision)
        required = {
            'idx', 'user_0', 'assistant_0', 'user_1', 'assistant_1', 'sphere'
        }
        missing = required.difference(split.column_names)
        if missing:
            raise ValueError(
                f'Earth-Gold is missing columns: {sorted(missing)}')
        rows = []
        for item in split:
            source_idx = str(item['idx'])
            reference = {
                key: item[key]
                for key in ('user_0', 'assistant_0', 'user_1', 'assistant_1',
                            'sphere')
            }
            dialogue = [
                {
                    'role': 'system',
                    'content': SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': GOLD_PROMPT + item['user_0']
                },
                {
                    'role': 'assistant',
                    'content': ''
                },
                {
                    'role': 'user',
                    'content': GOLD_PROMPT + item['user_1']
                },
                {
                    'role': 'assistant',
                    'content': ''
                },
            ]
            for repetition in range(repetitions):
                rows.append({
                    **dict(item),
                    'sample_id': f'earth_gold_train:{source_idx}',
                    'prediction_id':
                    f'earth_gold_train:{source_idx}:{repetition}',
                    'repetition': repetition,
                    'dialogue': copy.deepcopy(dialogue),
                    'reference': reference,
                })
        return Dataset.from_list(rows)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class _EarthSEJudgeMixin:

    def __init__(self,
                 judge_model_cfg: Optional[dict] = None,
                 judge_max_workers: int = 1,
                 judge_max_out_len: int = 4096,
                 **kwargs):
        super().__init__(**kwargs)
        self.judge_model_cfg = copy.deepcopy(judge_model_cfg)
        self.judge_max_workers = max(1, int(judge_max_workers))
        self.judge_max_out_len = judge_max_out_len
        self.judge_model = None

    def _build_judge_model(self):
        if self.judge_model is None:
            if self.judge_model_cfg is None:
                raise ValueError('EarthSE judge_model_cfg is required')
            self.judge_model = build_model_from_cfg(self.judge_model_cfg)
            if getattr(self.judge_model, 'is_api', False):
                self.judge_model.max_workers = self.judge_max_workers
                self.judge_model.tokens = BoundedSemaphore(
                    self.judge_max_workers)

    def _map_judgments(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        self._build_judge_model()
        messages = [[{
            'role': 'system',
            'content': SYSTEM_PROMPT
        }, {
            'role': 'user',
            'content': prompt
        }] for prompt in prompts]
        outputs = self.judge_model.generate(
            messages, max_out_len=self.judge_max_out_len, temperature=0)
        separator = getattr(self.judge_model, 'think_tag', None)
        return [earthse_extract_content(output, separator)
                for output in outputs]


@ICL_EVALUATORS.register_module()
class EarthSEEvaluator(_EarthSEJudgeMixin, BaseEvaluator):
    """Official QA exact-match, free-form judge, and grouped aggregation."""

    TASKS = (
        'knowledge_qa', 'fact_checking', 'analysis', 'calculation',
        'term_explanation', 'relationship_extraction', 'tool_usage',
        'literature_listing', 'dataset', 'experiment_design',
        'code_generation'
    )
    SPHERES = ('Hydrosphere', 'Biosphere', 'Lithosphere', 'Atmosphere',
               'Cryosphere')

    def __init__(self,
                 similarity_model_path: str =
                 'sentence-transformers/all-MiniLM-L6-v2',
                 similarity_model_revision: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.similarity_model_path = similarity_model_path
        self.similarity_model_revision = similarity_model_revision
        self.similarity_model = None

    def _build_similarity_model(self):
        if self.similarity_model is None:
            from sentence_transformers import SentenceTransformer
            kwargs = {}
            if self.similarity_model_revision:
                kwargs['revision'] = self.similarity_model_revision
            self.similarity_model = SentenceTransformer(
                self.similarity_model_path, **kwargs)

    def _similarities(self, pairs):
        if not pairs:
            return []
        self._build_similarity_model()
        from sentence_transformers import util
        values = []
        for reference, prediction in pairs:
            embedding_1 = self.similarity_model.encode(
                str(reference), convert_to_tensor=True)
            embedding_2 = self.similarity_model.encode(
                str(prediction), convert_to_tensor=True)
            values.append(
                util.pytorch_cos_sim(embedding_1, embedding_2).item())
        return values

    @staticmethod
    def build_free_form_judge_prompt(question, reference, prediction):
        return FREE_FORM_JUDGE_PROMPT.replace('<QUES>', str(question)).replace(
            '<ANS_A>', str(reference)).replace('<ANS_B>', str(prediction))

    def score(self, predictions, references, test_set):
        if len(predictions) != len(references):
            raise ValueError('predictions and references differ in length')
        free_indices = [
            i for i, row in enumerate(test_set)
            if row['question_type'] == 'free_form'
        ]
        judge_prompts = [
            self.build_free_form_judge_prompt(test_set[i]['question'],
                                              references[i], predictions[i])
            for i in free_indices
        ]
        judgments = self._map_judgments(judge_prompts) if judge_prompts else []
        similarities = self._similarities([(references[i], predictions[i])
                                           for i in free_indices])
        free_outputs = dict(zip(free_indices, zip(judgments, similarities)))

        details = []
        for i, (prediction, reference, row) in enumerate(
                zip(predictions, references, test_set)):
            detail = {
                'sample_id': row['sample_id'],
                'prediction': prediction,
                'reference': reference,
                'question_type': row['question_type'],
                'task': row['task'],
                'sphere': row['sphere'],
                'correct': prediction == reference,
            }
            if row.get('dataset_name'):
                detail['dataset_name'] = row['dataset_name']
            if i in free_outputs:
                judgment, similarity = free_outputs[i]
                # The official parser deliberately does not strip output.
                if judgment == 'A':
                    detail['win'] = 'reference_answer'
                elif judgment == 'B':
                    detail['win'] = 'llm_answer'
                detail['similarity'] = similarity
                detail['correct'] = (detail['correct']
                                     or detail.get('win') == 'llm_answer')
            details.append(detail)

        metrics = self.aggregate(details)
        metrics['details'] = details
        return metrics

    @staticmethod
    def aggregate(details):
        result = {}
        all_correct = [int(item['correct']) for item in details]
        result['ACC'] = _mean(all_correct) * 100
        for question_type in EarthSEDataset.SPLITS:
            selected = [
                item for item in details
                if item['question_type'] == question_type
            ]
            result[question_type] = _mean(
                [int(item['correct']) for item in selected]) * 100
            if question_type == 'free_form':
                result['free_form (Acc.)'] = result[question_type]
                result['free_form (SS)'] = _mean([
                    item['similarity'] for item in selected
                    if 'similarity' in item
                ])
        for field, values in (('task', EarthSEEvaluator.TASKS),
                              ('sphere', EarthSEEvaluator.SPHERES)):
            for value in values:
                selected = [item for item in details if item[field] == value]
                result[f'{field}/{value}'] = _mean(
                    [int(item['correct']) for item in selected]) * 100
        dataset_names = list(dict.fromkeys(
            item['dataset_name'] for item in details
            if item.get('dataset_name')))
        for dataset_name in dataset_names:
            for question_type in EarthSEDataset.SPLITS:
                selected = [
                    item for item in details
                    if item.get('dataset_name') == dataset_name
                    and item['question_type'] == question_type
                ]
                result[f'dataset/{dataset_name}/{question_type}'] = _mean(
                    [int(item['correct']) for item in selected]) * 100
        return result


@ICL_EVALUATORS.register_module()
class EarthSEGoldEvaluator(_EarthSEJudgeMixin, BaseEvaluator):
    """Official Earth-Gold retention, diversity, and SES implementation."""

    def __init__(self,
                 similarity_model_path: str =
                 'sentence-transformers/all-MiniLM-L6-v2',
                 similarity_model_revision: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.similarity_model_path = similarity_model_path
        self.similarity_model_revision = similarity_model_revision
        self.similarity_model = None

    def _build_similarity_model(self):
        if self.similarity_model is None:
            from sentence_transformers import SentenceTransformer
            kwargs = {}
            if self.similarity_model_revision:
                kwargs['revision'] = self.similarity_model_revision
            self.similarity_model = SentenceTransformer(
                self.similarity_model_path, **kwargs)

    @staticmethod
    def build_retention_judge_prompt(reference, dialogues):
        # These two dict literals intentionally preserve the official
        # duplicate-key behavior: only turn two remains after construction.
        dialogue_1 = {
            'user': reference['user_0'],
            'assistant': reference['assistant_0'],
            'user': reference['user_1'],
            'assistant': reference['assistant_1'],
        }
        dialogue_2 = [
            str({
                'user': d['user_0'],
                'assistant': d['assistant_0'],
                'user': d['user_1'],
                'assistant': d['assistant_1'],
            }) for d in dialogues
        ]
        return GOLD_JUDGE_PROMPT.replace('<Dialogue_1>', str(dialogue_1)) \
            .replace('<Dialogue_2>', '\n\n'.join(dialogue_2)) \
            .replace('<NUM>', str(len(dialogue_2) + 1))

    def _diversity(self, dialogues):
        self._build_similarity_model()
        from sentence_transformers import util
        import torch
        embeddings = [
            self.similarity_model.encode(
                d['assistant_0'] + d['assistant_1'], convert_to_tensor=True)
            for d in dialogues
        ]
        mean_embedding = torch.mean(torch.stack(embeddings), dim=0)
        similarity = _mean([
            util.pytorch_cos_sim(mean_embedding, embedding).item()
            for embedding in embeddings
        ])
        return 1 / (10 * max(abs(similarity) - 0.9, 0.01))

    def score(self, predictions, references, test_set):
        if not (len(predictions) == len(references) == len(test_set)):
            raise ValueError('Earth-Gold predictions are incomplete')
        groups = {}
        order = []
        for prediction, reference, row in zip(predictions, references,
                                              test_set):
            if not isinstance(prediction, list) or len(prediction) != 2:
                raise ValueError('Earth-Gold predictions require two turns')
            sample_id = row['sample_id']
            if sample_id not in groups:
                groups[sample_id] = {'reference': reference, 'answers': []}
                order.append(sample_id)
            groups[sample_id]['answers'].append({
                'user_0': reference['user_0'],
                'assistant_0': prediction[0],
                'user_1': reference['user_1'],
                'assistant_1': prediction[1],
            })
        for sample_id in order:
            if len(groups[sample_id]['answers']) != 3:
                raise ValueError(
                    f'Earth-Gold sample {sample_id} does not have 3 trials')

        prompts = [
            self.build_retention_judge_prompt(groups[sid]['reference'],
                                              groups[sid]['answers'])
            for sid in order
        ]
        judgments = self._map_judgments(prompts)
        logical_details = {}
        for sample_id, judgment in zip(order, judgments):
            group = groups[sample_id]
            diversity = self._diversity(group['answers'])
            detail = {'sample_id': sample_id, 'diversity': diversity}
            parsed = judgment.strip()
            if parsed in {'1', '2', '3', '4'}:
                retention = (int(parsed) - 1) / 3
                detail['retention_rate'] = retention
                detail['SES'] = retention * diversity
            logical_details[sample_id] = detail

        details = [copy.deepcopy(logical_details[row['sample_id']])
                   for row in test_set]
        logical = list(logical_details.values())
        result = {
            'retention_rate':
            _mean([x['retention_rate'] for x in logical
                   if 'retention_rate' in x]) * 100,
            'diversity': _mean([x['diversity'] for x in logical]),
            'SES': _mean([x['SES'] for x in logical if 'SES' in x]),
            'details': details,
        }
        return result


@ICL_INFERENCERS.register_module()
class EarthSEGoldInferencer(GenInferencer):
    """Two-turn native inferencer with Earth-Gold's per-turn temperatures."""

    def _generate_multiround(self, entry: List,
                             extra_gen_kwargs: dict) -> List[List[str]]:
        max_workers = self.batch_size
        turn_models = [self.model, self.model]
        if hasattr(self.model, 'temperature'):
            turn_models = [copy.copy(self.model), copy.copy(self.model)]
            turn_models[0].temperature = 0.6
            turn_models[1].temperature = 0
        generation_slots = []
        for chat in entry:
            generation_slots.append([
                i for i, message in enumerate(chat)
                if message.get('role') == 'assistant'
                and not message.get('content', '')
            ])

        def generate_turn(chat_idx, turn_idx):
            message_idx = generation_slots[chat_idx][turn_idx]
            history = copy.deepcopy(entry[chat_idx][:message_idx])
            # The pinned implementation prefixes both active questions, but
            # carries the raw first question in the second call's history.
            if turn_idx == 1:
                for message in history:
                    if (message.get('role') == 'user'
                            and message.get('content', '').startswith(
                                GOLD_PROMPT)):
                        message['content'] = message['content'][len(
                            GOLD_PROMPT):]
                        break
            temperature = 0.6 if turn_idx == 0 else 0
            output = turn_models[turn_idx].generate_from_template(
                [history],
                max_out_len=self.max_out_len,
                temperature=temperature,
                **extra_gen_kwargs,
            )[0]
            entry[chat_idx][message_idx]['content'] = output

        next_turn = [0] * len(entry)
        in_flight = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for chat_idx, slots in enumerate(generation_slots):
                if slots:
                    in_flight[executor.submit(generate_turn, chat_idx,
                                              0)] = chat_idx
                    next_turn[chat_idx] = 1
            while in_flight:
                done, _ = wait(set(in_flight), return_when=FIRST_COMPLETED)
                for future in done:
                    chat_idx = in_flight.pop(future)
                    future.result()
                    turn_idx = next_turn[chat_idx]
                    if turn_idx < len(generation_slots[chat_idx]):
                        in_flight[executor.submit(
                            generate_turn, chat_idx, turn_idx)] = chat_idx
                        next_turn[chat_idx] += 1

        return [[
            message['content'] for message in chat
            if message.get('role') == 'assistant'
        ] for chat in entry]
