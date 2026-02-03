import unittest

from opencompass.models.base import LMTemplateParser
from opencompass.utils.prompt import PromptList


class TestLMTemplateParser(unittest.TestCase):

    def setUp(self):
        self.parser = LMTemplateParser()
        self.prompt = PromptList([
            {
                'section': 'begin',
                'pos': 'begin'
            },
            'begin',
            {
                'role': 'SYSTEM',
                'fallback_role': 'HUMAN',
                'prompt': 'system msg'
            },
            {
                'section': 'ice',
                'pos': 'begin'
            },
            {
                'role': 'HUMAN',
                'prompt': 'U0'
            },
            {
                'role': 'BOT',
                'prompt': 'B0'
            },
            {
                'section': 'ice',
                'pos': 'end'
            },
            {
                'section': 'begin',
                'pos': 'end'
            },
            {
                'section': 'round',
                'pos': 'begin'
            },
            {
                'role': 'HUMAN',
                'prompt': 'U1',
                'end': '\n'
            },
            {
                'role': 'BOT',
                'prompt': 'B1'
            },
            {
                'role': 'HUMAN',
                'prompt': 'U2'
            },
            {
                'role': 'BOT',
                'prompt': 'B2'
            },
            {
                'section': 'round',
                'pos': 'end'
            },
            {
                'section': 'end',
                'pos': 'begin'
            },
            'end',
            {
                'section': 'end',
                'pos': 'end'
            },
        ])

    def test_parse_template_str_input(self):
        prompt = self.parser.parse_template('Hello, world!', mode='gen')
        self.assertEqual(prompt, 'Hello, world!')
        prompt = self.parser.parse_template('Hello, world!', mode='ppl')
        self.assertEqual(prompt, 'Hello, world!')

    def test_parse_template_list_input(self):
        prompt = self.parser.parse_template(['Hello', 'world'], mode='gen')
        self.assertEqual(prompt, ['Hello', 'world'])
        prompt = self.parser.parse_template(['Hello', 'world'], mode='ppl')
        self.assertEqual(prompt, ['Hello', 'world'])

    def test_parse_template_PromptList_input_no_meta_template(self):
        prompt = self.parser.parse_template(self.prompt, mode='gen')
        self.assertEqual(prompt,
                         'begin\nsystem msg\nU0\nB0\nU1\nB1\nU2\nB2\nend')
        prompt = self.parser.parse_template(self.prompt, mode='ppl')
        self.assertEqual(prompt,
                         'begin\nsystem msg\nU0\nB0\nU1\nB1\nU2\nB2\nend')

    def test_parse_template_PromptList_input_with_meta_template(self):
        # no SYSTEM role, early generation in THOUGHTS
        parser = LMTemplateParser(meta_template=dict(
            begin='meta instruction\n',
            round=[
                dict(role='HUMAN', begin='<|HUMAN|>:', end='<eoh>\n'),
                dict(role='THOUGHTS',
                     begin='<|Inner Thoughts|>:',
                     generate=True,
                     end='<eot>\n',
                     prompt='None'),
                dict(role='BOT', begin='<|BOT|>:', end='<eob>\n'),
            ],
            end='meta end',
        ))
        prompt = parser.parse_template(self.prompt, mode='gen')
        target = ('meta instruction\n'
                  'begin'
                  '<|HUMAN|>:system msg<eoh>\n'
                  '<|HUMAN|>:U0<eoh>\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:B0<eob>\n'
                  '<|HUMAN|>:U1\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:B1<eob>\n'
                  '<|HUMAN|>:U2<eoh>\n'
                  '<|Inner Thoughts|>:')
        self.assertEqual(prompt, target)
        prompt = parser.parse_template(self.prompt, mode='ppl')
        target = ('meta instruction\n'
                  'begin'
                  '<|HUMAN|>:system msg<eoh>\n'
                  '<|HUMAN|>:U0<eoh>\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:B0<eob>\n'
                  '<|HUMAN|>:U1\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:B1<eob>\n'
                  '<|HUMAN|>:U2<eoh>\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:B2<eob>\n'
                  'end'
                  'meta end')
        self.assertEqual(prompt, target)

        # no SYSTEM role, generation in BOT
        parser = LMTemplateParser(meta_template=dict(
            begin='meta instruction\n',
            round=[
                dict(role='HUMAN', begin='<|HUMAN|>:', end='<eoh>\n'),
                dict(role='THOUGHTS',
                     begin='<|Inner Thoughts|>:',
                     end='<eot>\n',
                     prompt='None'),
                dict(
                    role='BOT', begin='<|BOT|>:', end='<eob>\n',
                    generate=True),
            ],
            end='meta end',
        ))
        prompt = parser.parse_template(self.prompt, mode='gen')
        target = ('meta instruction\n'
                  'begin'
                  '<|HUMAN|>:system msg<eoh>\n'
                  '<|HUMAN|>:U0<eoh>\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:B0<eob>\n'
                  '<|HUMAN|>:U1\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:B1<eob>\n'
                  '<|HUMAN|>:U2<eoh>\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:')
        self.assertEqual(prompt, target)
        prompt = parser.parse_template(self.prompt, mode='ppl')
        target = ('meta instruction\n'
                  'begin'
                  '<|HUMAN|>:system msg<eoh>\n'
                  '<|HUMAN|>:U0<eoh>\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:B0<eob>\n'
                  '<|HUMAN|>:U1\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:B1<eob>\n'
                  '<|HUMAN|>:U2<eoh>\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:B2<eob>\n'
                  'end'
                  'meta end')
        self.assertEqual(prompt, target)

        # with SYSTEM role, generation in BOT
        parser = LMTemplateParser(meta_template=dict(
            begin='meta instruction\n',
            round=[
                dict(role='HUMAN', begin='<|HUMAN|>:', end='<eoh>\n'),
                dict(role='THOUGHTS',
                     begin='<|Inner Thoughts|>:',
                     end='<eot>\n',
                     prompt='None'),
                dict(
                    role='BOT', begin='<|BOT|>:', end='<eob>\n',
                    generate=True),
            ],
            end='meta end',
            reserved_roles=[
                dict(role='SYSTEM', begin='<|SYSTEM|>:', end='<eos>\n')
            ]))
        prompt = parser.parse_template(self.prompt, mode='gen')
        target = ('meta instruction\n'
                  'begin'
                  '<|SYSTEM|>:system msg<eos>\n'
                  '<|HUMAN|>:U0<eoh>\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:B0<eob>\n'
                  '<|HUMAN|>:U1\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:B1<eob>\n'
                  '<|HUMAN|>:U2<eoh>\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:')
        self.assertEqual(prompt, target)
        prompt = parser.parse_template(self.prompt, mode='ppl')
        target = ('meta instruction\n'
                  'begin'
                  '<|SYSTEM|>:system msg<eos>\n'
                  '<|HUMAN|>:U0<eoh>\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:B0<eob>\n'
                  '<|HUMAN|>:U1\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:B1<eob>\n'
                  '<|HUMAN|>:U2<eoh>\n'
                  '<|Inner Thoughts|>:None<eot>\n'
                  '<|BOT|>:B2<eob>\n'
                  'end'
                  'meta end')
        self.assertEqual(prompt, target)
