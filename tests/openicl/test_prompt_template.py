import unittest

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.utils.prompt import PromptList


class TestPromptTemplate(unittest.TestCase):

    def setUp(self) -> None:
        self.qa_template = dict(begin=[
            dict(role='SYSTEM', fallback_role='HUMAN', prompt='instruct'),
            '</E>',
        ],
                                round=[
                                    dict(role='HUMAN', prompt='{input}'),
                                    dict(role='BOT', prompt='Answer: {answer}')
                                ])
        self.multiround_qa_template = dict(round=[
            dict(role='HUMAN', prompt='{input}'),
            dict(role='BOT', prompt='A1', end='\n'),
            dict(role='HUMAN', prompt='Q1'),
            dict(role='BOT', prompt='A2', end='\n\n'),
            dict(role='HUMAN', prompt='Q2', begin='HUMAN:'),
            dict(role='BOT', prompt='Answer: {answer}')
        ])
        self.entry = {'input': 'Hello, how are you?', 'answer': 'Good.'}

    def test_init(self):
        template = 'Translate the following English text to French: {input}.'
        pt = PromptTemplate(template)

        self.assertEqual(pt.template, template)

    def test_generate_ice_item(self):
        # Test simple prompt
        template = 'Translate the following English text to French: {input}.'
        pt = PromptTemplate(template)
        label = None
        ice = pt.generate_ice_item(self.entry, label)

        self.assertEqual(ice,
                         ('Translate the following English text to French: '
                          'Hello, how are you?.'))

        # test meta prompt style
        pt = PromptTemplate(self.qa_template, ice_token='</E>')
        label = None
        ice = pt.generate_ice_item(self.entry, label)

        ice_target = PromptList([
            {
                'section': 'ice',
                'pos': 'begin'
            },
            dict(role='HUMAN', prompt='Hello, how are you?'),
            dict(role='BOT', prompt='Answer: Good.'),
            {
                'section': 'ice',
                'pos': 'end'
            },
        ])
        self.assertEqual(ice, ice_target)

        # test_multiround
        pt = PromptTemplate(self.multiround_qa_template, ice_token='</E>')
        label = None
        ice = pt.generate_ice_item(self.entry, label)

        ice_target = PromptList([
            {
                'section': 'ice',
                'pos': 'begin'
            },
            dict(role='HUMAN', prompt='Hello, how are you?'),
            dict(role='BOT', prompt='A1', end='\n'),
            dict(role='HUMAN', prompt='Q1'),
            dict(role='BOT', prompt='A2', end='\n\n'),
            dict(role='HUMAN', prompt='Q2', begin='HUMAN:'),
            dict(role='BOT', prompt='Answer: Good.'),
            {
                'section': 'ice',
                'pos': 'end'
            },
        ])
        self.assertEqual(ice, ice_target)

    def test_generate_label_prompt_item(self):
        # Test simple prompt
        template = ('</E> Translate the following English text to French: '
                    '{input}.')
        pt = PromptTemplate(template, ice_token='</E>')
        ice = 'ICE'
        label = None
        prompt = pt.generate_label_prompt_item(self.entry, ice, label)

        self.assertEqual(
            prompt, ('ICE Translate the following English text to French: '
                     'Hello, how are you?.'))

        ice = PromptList([
            {
                'section': 'ice',
                'pos': 'begin'
            },
            dict(role='HUMAN', prompt='h1'),
            dict(role='BOT', prompt='b1'),
            {
                'section': 'ice',
                'pos': 'end'
            },
        ])

        # test meta prompt style
        pt = PromptTemplate(self.qa_template, ice_token='</E>')
        label = None
        prompt = pt.generate_label_prompt_item(self.entry, ice, label)
        target = PromptList([
            {
                'section': 'begin',
                'pos': 'begin'
            },
            dict(role='SYSTEM', fallback_role='HUMAN', prompt='instruct'),
            {
                'section': 'ice',
                'pos': 'begin'
            },
            dict(role='HUMAN', prompt='h1'),
            dict(role='BOT', prompt='b1'),
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
            dict(role='HUMAN', prompt='Hello, how are you?'),
            dict(role='BOT', prompt='Answer: Good.'),
            {
                'section': 'round',
                'pos': 'end'
            },
        ])
        self.assertEqual(prompt, target)

        # test_multiround
        pt = PromptTemplate(self.multiround_qa_template, ice_token='</E>')
        label = None
        prompt = pt.generate_label_prompt_item(self.entry, ice, label)
        target = PromptList([
            {
                'section': 'round',
                'pos': 'begin'
            },
            dict(role='HUMAN', prompt='Hello, how are you?'),
            dict(role='BOT', prompt='A1', end='\n'),
            dict(role='HUMAN', prompt='Q1'),
            dict(role='BOT', prompt='A2', end='\n\n'),
            dict(role='HUMAN', prompt='Q2', begin='HUMAN:'),
            dict(role='BOT', prompt='Answer: Good.'),
            {
                'section': 'round',
                'pos': 'end'
            },
        ])
        self.assertEqual(prompt, target)

    def test_generate_item(self):
        # Test simple prompt
        template = 'Translate the following English text to French: {input}.'
        pt = PromptTemplate(template)
        item = pt.generate_item(self.entry)

        self.assertEqual(item,
                         ('Translate the following English text to French: '
                          'Hello, how are you?.'))

        ice = PromptList([
            {
                'section': 'ice',
                'pos': 'begin'
            },
            dict(role='HUMAN', prompt='h1'),
            dict(role='BOT', prompt='b1'),
            {
                'section': 'ice',
                'pos': 'end'
            },
        ])

        # test meta prompt (without system role)
        pt = PromptTemplate(self.qa_template, ice_token='</E>')
        prompt = pt.generate_item(self.entry, ice_field_replace_token=ice)
        target = PromptList([
            {
                'section': 'begin',
                'pos': 'begin'
            },
            dict(role='SYSTEM', fallback_role='HUMAN', prompt='instruct'),
            {
                'section': 'ice',
                'pos': 'begin'
            },
            dict(role='HUMAN', prompt='h1'),
            dict(role='BOT', prompt='b1'),
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
            dict(role='HUMAN', prompt='Hello, how are you?'),
            dict(role='BOT', prompt='Answer: Good.'),
            {
                'section': 'round',
                'pos': 'end'
            },
        ])
        self.assertEqual(prompt, target)

        pt = PromptTemplate(self.multiround_qa_template, ice_token='</E>')
        prompt = pt.generate_item(self.entry, ice_field_replace_token=ice)
        target = PromptList([
            {
                'section': 'round',
                'pos': 'begin'
            },
            dict(role='HUMAN', prompt='Hello, how are you?'),
            dict(role='BOT', prompt='A1', end='\n'),
            dict(role='HUMAN', prompt='Q1'),
            dict(role='BOT', prompt='A2', end='\n\n'),
            dict(role='HUMAN', prompt='Q2', begin='HUMAN:'),
            dict(role='BOT', prompt='Answer: Good.'),
            {
                'section': 'round',
                'pos': 'end'
            },
        ])
        self.assertEqual(prompt, target)
