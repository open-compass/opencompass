"""Check PPL scoring against unpadded, token-level causal LM losses."""

import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from opencompass.models.huggingface_above_v4_33 import HuggingFaceBaseModel


class TestHuggingFacePPL(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_dir = tempfile.TemporaryDirectory()
        vocab = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[UNK]': 3}
        words = 'The sky is blue capital of France Paris today and tomorrow'
        vocab.update({word: i + 4 for i, word in enumerate(words.split())})
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
                             max_position_embeddings=64,
                             pad_token_id=0,
                             bos_token_id=1,
                             eos_token_id=2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            LlamaForCausalLM(config).save_pretrained(cls.model_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.model_dir.cleanup()

    def setUp(self):
        self.model = HuggingFaceBaseModel(path=self.model_dir.name,
                                          max_seq_len=64,
                                          model_kwargs=dict(
                                              device_map='cpu',
                                              torch_dtype='torch.float32'))
        self.inputs = [
            'The sky is blue',
            'The capital of France is Paris today and tomorrow'
        ]

    def reference(self, text, mask_length=0):
        tokens = self.model.tokenizer(text, return_tensors='pt')
        if self.model.drop_middle:
            ids = tokens['input_ids']
            if ids.shape[1] > self.model.max_seq_len:
                half = self.model.max_seq_len // 2
                ids = torch.cat([ids[:, :half], ids[:, -half:]], dim=1)
            tokens = dict(input_ids=ids)
        with torch.no_grad():
            logits = self.model.model(**tokens).logits[0, :-1].float()
            losses = F.cross_entropy(logits,
                                     tokens['input_ids'][0, 1:],
                                     reduction='none')
        return losses[max(mask_length - 1, 0):].mean().item()

    def test_ppl_matches_reference_with_either_padding_side(self):
        for side in ['left', 'right']:
            self.model.tokenizer.padding_side = side
            for mask_lengths in [None, [0, 0], [2, 4]]:
                with self.subTest(side=side, mask_lengths=mask_lengths):
                    lengths = mask_lengths or [0, 0]
                    expected = [
                        self.reference(text, length)
                        for text, length in zip(self.inputs, lengths)
                    ]
                    with torch.no_grad():
                        actual = self.model.get_ppl(self.inputs, mask_lengths)
                        single = [
                            self.model.get_ppl(
                                [text],
                                None if mask_lengths is None else [length])[0]
                            for text, length in zip(self.inputs, lengths)
                        ]
                    np.testing.assert_allclose(actual, expected, rtol=1e-6)
                    np.testing.assert_allclose(single, expected, rtol=1e-6)

    def test_eos_tokens_are_scored_when_used_for_padding(self):
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        inputs = [text + ' [EOS]' for text in self.inputs]
        expected = [self.reference(text) for text in inputs]
        with torch.no_grad():
            actual = self.model.get_ppl(inputs)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_drop_middle_without_attention_mask(self):
        self.model.drop_middle = True
        self.model.max_seq_len = 6
        inputs = [self.inputs[1]]
        for mask_length in [0, 2]:
            with self.subTest(mask_length=mask_length):
                expected = [self.reference(inputs[0], mask_length)]
                with torch.no_grad():
                    actual = self.model.get_ppl(inputs, [mask_length])
                np.testing.assert_allclose(actual, expected, rtol=1e-6)
