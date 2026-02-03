import unittest

from opencompass.models.base_api import APITemplateParser
from opencompass.utils.prompt import PromptList


class TestAPITemplateParser(unittest.TestCase):

    def setUp(self):
        self.parser = APITemplateParser()
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
                'prompt': 'U1'
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
        parser = APITemplateParser(meta_template=dict(round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True)
        ], ))
        with self.assertWarns(Warning):
            prompt = parser.parse_template(self.prompt, mode='gen')
            self.assertEqual(
                prompt,
                PromptList([
                    {
                        'role': 'HUMAN',
                        'prompt': 'system msg\nU0'
                    },
                    {
                        'role': 'BOT',
                        'prompt': 'B0'
                    },
                    {
                        'role': 'HUMAN',
                        'prompt': 'U1'
                    },
                    {
                        'role': 'BOT',
                        'prompt': 'B1'
                    },
                    {
                        'role': 'HUMAN',
                        'prompt': 'U2'
                    },
                ]))
        with self.assertWarns(Warning):
            prompt = parser.parse_template(self.prompt, mode='ppl')
            self.assertEqual(
                prompt,
                PromptList([
                    {
                        'role': 'HUMAN',
                        'prompt': 'system msg\nU0'
                    },
                    {
                        'role': 'BOT',
                        'prompt': 'B0'
                    },
                    {
                        'role': 'HUMAN',
                        'prompt': 'U1'
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
                ]))

        parser = APITemplateParser(meta_template=dict(
            round=[
                dict(role='HUMAN', api_role='HUMAN'),
                dict(role='BOT', api_role='BOT', generate=True)
            ],
            reserved_roles=[
                dict(role='SYSTEM', api_role='SYSTEM'),
            ],
        ))
        with self.assertWarns(Warning):
            prompt = parser.parse_template(self.prompt, mode='gen')
            self.assertEqual(
                prompt,
                PromptList([
                    {
                        'role': 'SYSTEM',
                        'prompt': 'system msg'
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
                        'role': 'HUMAN',
                        'prompt': 'U1'
                    },
                    {
                        'role': 'BOT',
                        'prompt': 'B1'
                    },
                    {
                        'role': 'HUMAN',
                        'prompt': 'U2'
                    },
                ]))
        with self.assertWarns(Warning):
            prompt = parser.parse_template(self.prompt, mode='ppl')
            self.assertEqual(
                prompt,
                PromptList([
                    {
                        'role': 'SYSTEM',
                        'prompt': 'system msg'
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
                        'role': 'HUMAN',
                        'prompt': 'U1'
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
                ]))
