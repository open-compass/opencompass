import unittest

from opencompass.utils.prompt import PromptList


class TestPromptList(unittest.TestCase):

    def test_initialization(self):
        pl = PromptList()
        self.assertEqual(pl, [])

        pl = PromptList(['test', '123'])
        self.assertEqual(pl, ['test', '123'])

    def test_format(self):
        pl = PromptList(['hi {a}{b}', {'prompt': 'hey {a}!'}, '123'])
        new_pl = pl.format(a=1, b=2, c=3)
        self.assertEqual(new_pl, ['hi 12', {'prompt': 'hey 1!'}, '123'])

        new_pl = pl.format(b=2)
        self.assertEqual(new_pl, ['hi {a}2', {'prompt': 'hey {a}!'}, '123'])

        new_pl = pl.format(d=1)
        self.assertEqual(new_pl, ['hi {a}{b}', {'prompt': 'hey {a}!'}, '123'])

    def test_replace(self):
        pl = PromptList(['hello world', {'prompt': 'hello world'}, '123'])
        new_pl = pl.replace('world', 'there')
        self.assertEqual(new_pl,
                         ['hello there', {
                             'prompt': 'hello there'
                         }, '123'])

        new_pl = pl.replace('123', PromptList(['p', {'role': 'BOT'}]))
        self.assertEqual(
            new_pl,
            ['hello world', {
                'prompt': 'hello world'
            }, 'p', {
                'role': 'BOT'
            }])

        new_pl = pl.replace('2', PromptList(['p', {'role': 'BOT'}]))
        self.assertEqual(new_pl, [
            'hello world', {
                'prompt': 'hello world'
            }, '1', 'p', {
                'role': 'BOT'
            }, '3'
        ])

        with self.assertRaises(TypeError):
            new_pl = pl.replace('world', PromptList(['new', 'world']))

    def test_add(self):
        pl = PromptList(['hello'])
        new_pl = pl + ' world'
        self.assertEqual(new_pl, ['hello', ' world'])

        pl2 = PromptList([' world'])
        new_pl = pl + pl2
        self.assertEqual(new_pl, ['hello', ' world'])

        new_pl = 'hi, ' + pl
        self.assertEqual(new_pl, ['hi, ', 'hello'])

        pl += '!'
        self.assertEqual(pl, ['hello', '!'])

    def test_str(self):
        pl = PromptList(['hello', ' world', {'prompt': '!'}])
        self.assertEqual(str(pl), 'hello world!')

        with self.assertRaises(TypeError):
            pl = PromptList(['hello', ' world', 123])
            str(pl)
