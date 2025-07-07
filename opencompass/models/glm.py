import re
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


@MODELS.register_module(name=['GLM-130B'])
class GLM130B(BaseModel):

    def __init__(self,
                 pkg_root: str,
                 ckpt_path: str,
                 tokenizer_only: bool = False,
                 meta_template: Optional[Dict] = None,
                 **kwargs):
        assert not tokenizer_only, 'LLama does not support tokenizer only mode'
        self.pkg_root = pkg_root
        self.ckpt_path = ckpt_path
        self._load_model(**kwargs)

        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']

    def _load_model(self, **kwargs):
        import sys
        sys.path.insert(0, self.pkg_root)
        from argparse import Namespace

        from evaluation.model import ModelForEvaluation, batch_filling_sequence
        from generate import get_masks_and_position_ids
        from generation import BaseStrategy, BeamSearchStrategy
        from initialize import initialize_model_and_tokenizer
        from SwissArmyTransformer import get_args

        self.get_masks_and_position_ids = get_masks_and_position_ids
        self.batch_filling_sequence = batch_filling_sequence

        kwargs = {
            'bminf': False,
            'bminf_memory_limit': 20,
            'quantization_bit_width': None,
            'from_quantized_checkpoint': False,
            'sequential_initialization': False,
            'sampling_strategy': 'BaseStrategy',
            'min_gen_length': 0,
            'print_all_beams': False,
            **kwargs,
        }

        args_list = [
            ['--seed', '1234'],
            ['--mode', 'inference'],
            ['--out-seq-length', '256'],
            ['--num-beams', '4'],
            ['--length-penalty', '1.0'],
            ['--no-repeat-ngram-size', '3'],
            ['--temperature', '1.0'],
            ['--top_k', '0'],
            ['--top_p', '0'],
            ['--output-path', 'samples'],
            ['--model-parallel-size', '8'],
            ['--num-layers', '70'],
            ['--hidden-size', '12288'],
            ['--inner-hidden-size', '32768'],
            ['--vocab-size', '150528'],
            ['--num-attention-heads', '96'],
            ['--max-sequence-length', '2048'],
            ['--tokenizer-type', 'icetk-glm-130B'],
            ['--layernorm-order', 'post'],
            ['--load', self.ckpt_path],
            ['--skip-init'],
            ['--fp16'],
            ['--input-source', 'interactive'],
        ]  # Come from the default initialize arguments of official repo
        args = get_args(sum(args_list, []))
        args = Namespace(**vars(args), **kwargs)
        args.do_train = False
        self.args = args

        model, tokenizer = initialize_model_and_tokenizer(args)
        self.model = model
        self.model_for_eval = ModelForEvaluation(model)
        self.tokenizer = tokenizer
        self.device = args.device

        end_tokens = [
            tokenizer.get_command('eop'),
            tokenizer.get_command('eos')
        ]
        if args.sampling_strategy == 'BaseStrategy':
            self.strategy = BaseStrategy(batch_size=1,
                                         temperature=args.temperature,
                                         top_k=args.top_k,
                                         top_p=args.top_p,
                                         end_tokens=end_tokens)
        elif args.sampling_strategy == 'BeamSearchStrategy':
            self.strategy = BeamSearchStrategy(
                1,
                args.num_beams,
                length_penalty=args.length_penalty,
                consider_end=True,
                end_tokens=end_tokens,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                min_gen_length=args.min_gen_length,
            )
        else:
            raise ValueError(f'unknown strategy {args.sampling_strategy}')

        sys.path.pop(0)

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.tokenize(prompt))

    def choice(self, inputs, choices):
        import sys
        sys.path.insert(0, self.pkg_root)
        from unittest.mock import MagicMock

        from evaluation.dataset import MultiChoiceTaskDataset
        sys.path.pop(0)

        choice_tokens = [self.tokenizer.tokenize(item) for item in choices]
        is_single_token = all(len(token) == 1 for token in choice_tokens)

        data_items = []
        mock_dataset = MagicMock(is_single_token=is_single_token)
        from mmengine.dist import is_main_process
        for text in inputs:
            if is_main_process():
                print(f"\033[92m'text'\033[0m: {text}")
            data_item = MultiChoiceTaskDataset.build_multiple_choice_sample(
                text=self.tokenizer.tokenize(text),
                #  text=self.tokenizer.tokenize(text) + [20019],
                choices=[self.tokenizer.tokenize(item) for item in choices],
                is_single_token=is_single_token,
            )
            data_items.append(data_item)
        batch = MultiChoiceTaskDataset.collate_fn(mock_dataset, data_items)

        log_probs = self.model_for_eval.cond_log_prob(batch)

        answers = []
        for log_prob in zip(log_probs):
            answers.append(choices[np.argmax(log_prob).item()])

        return answers

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if isinstance(inputs, list):
            return sum((self.generate(raw_text, max_out_len)
                        for raw_text in inputs), [])
        else:
            raw_text = inputs

        from mmengine.dist import is_main_process
        if is_main_process():
            print(f"\033[92m'raw_text'\033[0m: \n{raw_text}")

        # add MASK
        generation_mask = '[gMASK]'
        if '[MASK]' in raw_text:
            generation_mask = '[MASK]'
        elif '[sMASK]' in raw_text:
            generation_mask = '[sMASK]'
        use_gmask = '[MASK]' not in raw_text and '[sMASK]' not in raw_text

        mask_pattern = r'\[[sg]?MASK\]'
        text_list = re.split(mask_pattern, raw_text)
        pattern_list = re.compile(mask_pattern).findall(raw_text)
        seq = []
        for i in range(len(pattern_list)):
            pattern = pattern_list[i]
            sub_text = text_list[i]
            seq.extend(self.tokenizer.tokenize(sub_text))
            seq.append(self.tokenizer.get_command(pattern))

        seq.extend(self.tokenizer.tokenize(text_list[-1]))
        prompt_token_length = len(seq)

        if 'MASK]' not in raw_text:
            seq += [self.tokenizer.get_command(generation_mask)]
            raw_text += ' ' + generation_mask
        if not raw_text.endswith('MASK]'):
            seq = seq + [self.tokenizer.get_command('eos')]
        if len(seq) > self.args.max_sequence_length:
            raise ValueError('text too long.')

        # generation
        output_list = [seq]
        if self.args.sampling_strategy == 'BeamSearchStrategy':
            num_output = self.args.num_beams
        else:
            num_output = 1
        last_pos = [0] * num_output

        # continually detect the first mark position
        while True:
            seq = output_list[0]
            # detect mask position
            mask_token = self.tokenizer.get_command(generation_mask)
            if mask_token not in seq:
                break
            mask_position = seq.index(mask_token)

            output_list = []

            input_seq = torch.cuda.LongTensor(
                [seq + [self.tokenizer.get_command('sop')]],
                device=self.device,
            )
            output, _ = self.batch_filling_sequence(
                self.model,
                input_seq,
                torch.cuda.LongTensor([input_seq.shape[-1]],
                                      device=self.device),
                strategy=self.strategy,
                get_masks_and_position_ids=partial(
                    self.get_masks_and_position_ids,
                    mask_position=mask_position,
                    max_gen_length=max_out_len,
                    gmask=use_gmask,
                ),
            )
            if isinstance(output, torch.Tensor):  # different strategies
                output = output.tolist()
            output = output[0]  # batch_size = 1
            output_list.extend(output)

            # clip -1s and fill back generated things into seq
            for i in range(len(output_list)):
                output = output_list[i].tolist() if isinstance(
                    output_list[i], torch.Tensor) else output_list[i]
                try:
                    unfinished = output.index(-1)
                except ValueError:
                    unfinished = len(output)
                if output[unfinished - 1] in self.strategy.end_tokens:
                    unfinished -= 1
                bog = output.index(self.tokenizer.get_command('sop'))

                last_pos[i] = mask_position + unfinished - (bog + 1)
                output_list[i] = output[:mask_position] + output[
                    bog + 1:unfinished] + output[mask_position + 1:bog]

        # Select the best answer
        output = output_list[0]
        if output[-1] == self.tokenizer.get_command('eos'):
            output = output[:-1]

        # Avoid generate out-of-range id, replace to unk
        output = np.array(output)
        output[output < 20000] = 20000
        output = output.tolist()
        answer = self.tokenizer.detokenize(output[prompt_token_length:])
        if is_main_process():
            print(f"\033[92m'answer'\033[0m: \n{answer}")

        return [answer]

    def get_logits(self, inputs: List[str]):
        mask_id = self.tokenizer.get_command('[MASK]')
        sop_id = self.tokenizer.get_command('sop')

        tokens = []
        targets = []
        position_ids = []
        attn_masks = []
        from mmengine.dist import is_main_process
        for raw_text in inputs:
            mask_pattern = r'\[MASK\]'
            text_list = re.split(mask_pattern, raw_text, 1)

            token = sum([
                self.tokenizer.tokenize(text_list[0]),
                [mask_id, sop_id],
                self.tokenizer.tokenize(text_list[1]),
            ], [])[:-1]
            target = sum([
                self.tokenizer.tokenize(text_list[0]),
                [mask_id],
                self.tokenizer.tokenize(text_list[1]),
            ], [])
            if is_main_process():
                print(f"\033[92m'raw_text'\033[0m: {raw_text}")
                print(f"\033[92m'token'\033[0m: {token}")

            seq_length = len(token)

            attn_mask = np.ones((seq_length, seq_length), dtype=np.int64)

            tokens.append(np.array(token, dtype=np.int64))
            targets.append(np.array(target, dtype=np.int64))
            position_ids.append(np.arange(0, seq_length, dtype=np.int64))
            attn_masks.append(attn_mask)

        TILE = 32
        length_to_pad = (max(map(len, tokens)) + TILE - 1) // TILE * TILE
        token_batch, target_batch, position_id_batch, attention_mask_batch = [], [], [], []  # noqa: E501
        for token, target, position_id, attn_mask in zip(
                tokens, targets, position_ids, attn_masks):
            attn_mask = np.pad(
                attn_mask,
                pad_width=((0, length_to_pad - len(token)), ),
                mode='constant',
                constant_values=0,
            )
            token = np.concatenate(
                (token, np.zeros(length_to_pad - len(token), dtype=np.int64)))
            target = np.concatenate((target,
                                     np.full(length_to_pad - len(target),
                                             -1,
                                             dtype=np.int64)))
            position_id = np.concatenate(
                (position_id,
                 np.zeros(length_to_pad - len(position_id), dtype=np.int64)))

            token_batch.append(token)
            target_batch.append(target)
            position_id_batch.append(position_id)
            attention_mask_batch.append(attn_mask)

        token_batch = torch.tensor(np.array(token_batch),
                                   dtype=torch.int64).to(self.device)
        target_batch = torch.tensor(np.array(target_batch),
                                    dtype=torch.int64).to(self.device)
        position_id_batch = torch.tensor(np.array(position_id_batch),
                                         dtype=torch.int64).to(self.device)
        attention_mask_batch = (torch.tensor(
            np.array(attention_mask_batch), dtype=torch.int64) < 0.5).to(
                self.device).bool().unsqueeze(1)

        logits, *out_per_layers = self.model(token_batch,
                                             position_id_batch,
                                             attention_mask_batch,
                                             log_attention_weights=None)
        if is_main_process():
            print(f"\033[92m'target_batch'\033[0m: {target_batch}")

        return logits, target_batch

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        logits, targets = self.get_logits(inputs)

        loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        loss = loss_fn(logits.view(-1, logits.size(-1)),
                       targets.view(-1)).view(targets.size())
        from mmengine.dist import is_main_process
        if is_main_process():
            print(f"\033[92m'loss'\033[0m: {loss}")

        if mask_length is not None:
            mask = torch.zeros_like(targets)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (targets != -1).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        if is_main_process():
            print(f"\033[92m'lens'\033[0m: {lens}")
            print(f"\033[92m'ce_loss'\033[0m: {ce_loss}")
        return ce_loss
