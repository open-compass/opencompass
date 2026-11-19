"""Check continuation scores against a causal LM's masked-label loss."""

import tempfile
import unittest

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from opencompass.models.huggingface_above_v4_33 import HuggingFaceBaseModel


class TestHuggingFaceLoglikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_dir = tempfile.TemporaryDirectory()
        vocab = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[UNK]': 3}
        words = 'Alice asked Bob for help because was confused . The trophy did not fit in the suitcase it too large'
        vocab.update({
            word: i + 4
            for i, word in enumerate(dict.fromkeys(words.split()))
        })
        tokenizer = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(single='[BOS] $A',
                                                      special_tokens=[('[BOS]',
                                                                       1)])
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token='[BOS]',
            eos_token='[EOS]',
            pad_token='[PAD]',
            unk_token='[UNK]',
            model_input_names=['input_ids', 'attention_mask'])
        tokenizer.save_pretrained(cls.model_dir.name)
        config = LlamaConfig(vocab_size=len(vocab),
                             hidden_size=16,
                             intermediate_size=32,
                             num_hidden_layers=1,
                             num_attention_heads=2,
                             num_key_value_heads=2,
                             max_position_embeddings=128,
                             bos_token_id=1,
                             eos_token_id=2,
                             pad_token_id=0)
        torch.manual_seed(7)
        LlamaForCausalLM(config).save_pretrained(cls.model_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.model_dir.cleanup()

    def setUp(self):
        self.model = HuggingFaceBaseModel(path=self.model_dir.name,
                                          max_seq_len=128,
                                          model_kwargs=dict(
                                              device_map='cpu',
                                              torch_dtype=torch.float32))

    def reference(self, context, continuation, keep=None):
        encoded = self.model.tokenizer(context + continuation,
                                       return_tensors='pt',
                                       return_offsets_mapping=True,
                                       truncation=not self.model.drop_middle,
                                       max_length=self.model.max_seq_len)
        ids = encoded['input_ids']
        labels = ids.clone()
        labels[encoded['offset_mapping'][:, :, 1] <= len(context)] = -100
        if keep is not None:
            ids, labels = ids[:, keep], labels[:, keep]
        target_count = (labels[:, 1:] != -100).sum().item()
        if not target_count:
            return 0.0
        with torch.no_grad():
            return -self.model.model(input_ids=ids,
                                     labels=labels).loss.item() * target_count

    def test_scores_only_the_continuation_and_sums_token_logprobs(self):
        pairs = [
            ('Alice asked Bob for help because Alice', ' was confused.'),
            ('The trophy did not fit in the suitcase because it',
             ' was too large.'),
        ]
        expected = [self.reference(*pair) for pair in pairs]
        for padding_side in ('left', 'right'):
            with self.subTest(padding_side=padding_side):
                self.model.tokenizer.padding_side = padding_side
                with torch.no_grad():
                    actual = self.model.get_loglikelihood([
                        context + continuation
                        for context, continuation in pairs
                    ], [continuation for _, continuation in pairs])
                np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_empty_continuation_has_zero_loglikelihood(self):
        with torch.no_grad():
            actual = self.model.get_loglikelihood(['Alice asked Bob'], [''])
        np.testing.assert_array_equal(actual, [0.0])

    def test_truncation_preserves_the_continuation_mask(self):
        context = 'The trophy did not fit in the suitcase because it'
        continuation = ' was too large.'
        self.model.max_seq_len = 8
        self.model.tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
            single='[BOS] $A [EOS]',
            special_tokens=[('[BOS]', 1), ('[EOS]', 2)])
        for side in ('left', 'right'):
            with self.subTest(truncation_side=side):
                self.model.tokenizer.truncation_side = side
                with torch.no_grad():
                    actual = self.model.get_loglikelihood(
                        [context + continuation], [continuation])
                np.testing.assert_allclose(
                    actual, [self.reference(context, continuation)], rtol=1e-6)

    def test_middle_truncation_preserves_the_continuation_mask(self):
        context = 'The trophy did not fit in the suitcase because it'
        continuation = ' was too large.'
        self.model.drop_middle = True
        for length, keep in [(8, list(range(4)) + list(range(-4, 0))),
                             (9, list(range(4)) + list(range(-5, 0))),
                             (1, [-1])]:
            with self.subTest(max_seq_len=length):
                self.model.max_seq_len = length
                with torch.no_grad():
                    actual = self.model.get_loglikelihood(
                        [context + continuation], [continuation])
                np.testing.assert_allclose(
                    actual, [self.reference(context, continuation, keep)],
                    rtol=1e-6)

    def test_token_crossing_the_context_boundary_is_scored(self):
        pieces = [
            '[PAD]', '[BOS]', '[EOS]', '[UNK]', 'A', 'l', 'i', 'c', 'e', 'w',
            'a', 's', 'o', 'n', 'f', 'u', 'd', '.', 'co', 'con', 'conf',
            'confu', 'confus', 'confuse', 'confused'
        ]
        tokenizer = Tokenizer(
            BPE({
                piece: i
                for i, piece in enumerate(pieces)
            },
                merges=[('c', 'o'), ('co', 'n'), ('con', 'f'), ('conf', 'u'),
                        ('confu', 's'), ('confus', 'e'), ('confuse', 'd')],
                unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(single='[BOS] $A',
                                                      special_tokens=[('[BOS]',
                                                                       1)])
        self.model.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token='[PAD]',
            bos_token='[BOS]',
            eos_token='[EOS]',
            unk_token='[UNK]')
        self.model.model.resize_token_embeddings(len(pieces), mean_resizing=False)
        context, continuation = 'Alice was con', 'fused.'
        with torch.no_grad():
            actual = self.model.get_loglikelihood([context + continuation],
                                                  [continuation])
        np.testing.assert_allclose(actual,
                                   [self.reference(context, continuation)],
                                   rtol=1e-6)

    def test_eos_in_continuation_is_scored_even_when_used_as_padding(self):
        self.model.tokenizer.pad_token = '[EOS]'
        pairs = [('Alice was', ' confused.[EOS]'), ('Bob', ' was confused.')]
        with torch.no_grad():
            actual = self.model.get_loglikelihood(
                [context + continuation for context, continuation in pairs],
                [continuation for _, continuation in pairs])
        np.testing.assert_allclose(actual,
                                   [self.reference(*pair) for pair in pairs],
                                   rtol=1e-6)
