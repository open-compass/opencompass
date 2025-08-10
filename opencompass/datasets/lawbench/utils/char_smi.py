### Copy from https://github.com/iqiyi/FASPell ###

"""
Requirements:
 - java (required only if tree edit distance is used)
 - numpy
"""
import numpy as np
from subprocess import Popen, PIPE, STDOUT
import os
import argparse

IDCS = {'\u2ff0': 2,  # 12 ideographic description characters and their capacity of son nodes
        '\u2ff1': 2,
        '\u2ff2': 3,
        '\u2ff3': 3,
        '\u2ff4': 2,
        '\u2ff5': 2,
        '\u2ff6': 2,
        '\u2ff7': 2,
        '\u2ff8': 2,
        '\u2ff9': 2,
        '\u2ffa': 2,
        '\u2ffb': 2, }

PINYIN = {'ā': ['a', 1], 'á': ['a', 2], 'ǎ': ['a', 3], 'à': ['a', 4],
          'ē': ['e', 1], 'é': ['e', 2], 'ě': ['e', 3], 'è': ['e', 4],
          'ī': ['i', 1], 'í': ['i', 2], 'ǐ': ['i', 3], 'ì': ['i', 4],
          'ō': ['o', 1], 'ó': ['o', 2], 'ǒ': ['o', 3], 'ò': ['o', 4],
          'ū': ['u', 1], 'ú': ['u', 2], 'ǔ': ['u', 3], 'ù': ['u', 4],
          'ǖ': ['ü', 1], 'ǘ': ['ü', 2], 'ǚ': ['ü', 3], 'ǜ': ['ü', 4],
          '': ['m', 2], 'ń': ['n', 2], 'ň': ['n', 3], 'ǹ': ['n', 4],
          }

# APTED_JAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'apted.jar')
APTED_JAR_PATH = 'apted.jar'


def tree_edit_distance(tree_a, tree_b):
    """
    We use APTED algorithm proposed by M. Pawlik and N. Augsten
    github link: https://github.com/DatabaseGroup/apted
    """
    p = Popen(['java', '-jar', APTED_JAR_PATH, '-t', tree_a, tree_b], stdout=PIPE, stderr=STDOUT)

    res = [line for line in p.stdout]
    res = res[0]
    res = res.strip()
    res = float(res)

    return res


def edit_distance(string_a, string_b, name='Levenshtein'):
    """
    >>> edit_distance('abcde', 'avbcude')
    2
    >>> edit_distance(['至', '刂'], ['亻', '至', '刂'])
    1
    >>> edit_distance('fang', 'qwe')
    4
    >>> edit_distance('fang', 'hen')
    3
    """
    size_x = len(string_a) + 1
    size_y = len(string_b) + 1
    matrix = np.zeros((size_x, size_y), dtype=int)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if string_a[x - 1] == string_b[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                if name == 'Levenshtein':
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1] + 1,
                        matrix[x, y - 1] + 1
                    )
                else:  # Canonical
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1] + 2,
                        matrix[x, y - 1] + 1
                    )

    return matrix[size_x - 1, size_y - 1]


class CharFuncs(object):
    def __init__(self, char_meta_fname):
        self.data = self.load_char_meta(char_meta_fname)
        self.char_dict = dict([(c, 0) for c in self.data])

        self.safe = {'\u2ff0': 'A',
                     # to eliminate the bug that, in Windows CMD, char ⿻ and ⿵ are encoded to be the same.
                     '\u2ff1': 'B',
                     '\u2ff2': 'C',
                     '\u2ff3': 'D',
                     '\u2ff4': 'E',
                     '\u2ff5': 'F',
                     '\u2ff6': 'G',
                     '\u2ff7': 'H',
                     '\u2ff8': 'I',
                     '\u2ff9': 'J',
                     '\u2ffa': 'L',
                     '\u2ffb': 'M', }

    @staticmethod
    def load_char_meta(fname):
        data = {}
        f = open(fname, 'r', encoding='utf-8')
        for line in f:
            items = line.strip().split('\t')
            code_point = items[0]
            char = items[1]
            pronunciation = items[2]
            decompositions = items[3:]
            assert char not in data
            data[char] = {"code_point": code_point, "pronunciation": pronunciation, "decompositions": decompositions}
        return data

    def shape_distance(self, char1, char2, safe=True, as_tree=False):
        """
        >>> c = CharFuncs('data/char_meta.txt')
        >>> c.shape_distance('田', '由')
        1
        >>> c.shape_distance('牛', '午')
        1
        """
        assert char1 in self.data
        assert char2 in self.data

        def safe_encode(decomp):
            tree = ''
            for c in string_to_tree(decomp):
                if c not in self.safe:
                    tree += c
                else:
                    tree += self.safe[c]
            return tree

        def safe_encode_string(decomp):
            tree = ''
            for c in decomp:
                if c not in self.safe:
                    tree += c
                else:
                    tree += self.safe[c]
            return tree

        decomps_1 = self.data[char1]["decompositions"]
        decomps_2 = self.data[char2]["decompositions"]

        distance = 1e5
        if as_tree:
            for decomp1 in decomps_1:
                for decomp2 in decomps_2:
                    if not safe:
                        ted = tree_edit_distance(string_to_tree(decomp1), string_to_tree(decomp2))
                    else:
                        ted = tree_edit_distance(safe_encode(decomp1), safe_encode(decomp2))
                        distance = min(distance, ted)
        else:
            for decomp1 in decomps_1:
                for decomp2 in decomps_2:
                    if not safe:
                        ed = edit_distance(decomp1, decomp2)
                    else:
                        ed = edit_distance(safe_encode_string(decomp1), safe_encode_string(decomp2))
                    distance = min(distance, ed)

        return distance

    def pronunciation_distance(self, char1, char2):
        """
        >>> c = CharFuncs('data/char_meta.txt')
        >>> c.pronunciation_distance('田', '由')
        3.4
        >>> c.pronunciation_distance('牛', '午')
        2.6
        """
        assert char1 in self.data
        assert char2 in self.data
        pronunciations1 = self.data[char1]["pronunciation"]
        pronunciations2 = self.data[char2]["pronunciation"]

        if pronunciations1[0] == 'null' or pronunciations2 == 'null':
            return 0.0
        else:

            pronunciations1 = pronunciations1.split(';')  # separate by lan
            pronunciations2 = pronunciations2.split(';')  # separate by lan

            distance = 0.0
            count = 0
            for pron_lan1, pron_lan2 in zip(pronunciations1, pronunciations2):
                if (pron_lan1 == 'null') or (pron_lan2 == 'null'):
                    pass
                else:
                    distance_lan = 1e5
                    for p1 in pron_lan1.split(','):
                        for p2 in pron_lan2.split(','):
                            distance_lan = min(distance_lan, edit_distance(p1, p2))
                    distance += distance_lan
                    count += 1

            return distance / count

    @staticmethod
    def load_dict(fname):
        data = {}
        f = open(fname, 'r', encoding='utf-8')
        for line in f:
            char, freq = line.strip().split('\t')
            assert char not in data
            data[char] = freq

        return data

    def similarity(self, char1, char2, weights=(0.8, 0.2, 0.0), as_tree=False):
        """
        this function returns weighted similarity. When used in FASPell, each weight can only be 0 or 1.
        """

        # assert char1 in self.char_dict
        # assert char2 in self.char_dict
        shape_w, sound_w, freq_w = weights

        if char1 in self.char_dict and char2 in self.char_dict:

            shape_sim = self.shape_similarity(char1, char2, as_tree=as_tree)
            sound_sim = self.pronunciation_similarity(char1, char2)
            freq_sim = 1.0 - self.char_dict[char2] / len(self.char_dict)

            return shape_sim * shape_w + sound_sim * sound_w + freq_sim * freq_w
        else:
            return 0.0

    def shape_similarity(self, char1, char2, safe=True, as_tree=False):
        """
        >>> c = CharFuncs('data/char_meta.txt')
        >>> c.shape_similarity('牛', '午')
        0.8571428571428572
        >>> c.shape_similarity('田', '由')
        0.8888888888888888
        """
        assert char1 in self.data
        assert char2 in self.data

        def safe_encode(decomp):
            tree = ''
            for c in string_to_tree(decomp):
                if c not in self.safe:
                    tree += c
                else:
                    tree += self.safe[c]
            return tree

        def safe_encode_string(decomp):
            tree = ''
            for c in decomp:
                if c not in self.safe:
                    tree += c
                else:
                    tree += self.safe[c]
            return tree

        decomps_1 = self.data[char1]["decompositions"]
        decomps_2 = self.data[char2]["decompositions"]

        similarity = 0.0
        if as_tree:
            for decomp1 in decomps_1:
                for decomp2 in decomps_2:
                    if not safe:
                        ted = tree_edit_distance(string_to_tree(decomp1), string_to_tree(decomp2))
                    else:
                        ted = tree_edit_distance(safe_encode(decomp1), safe_encode(decomp2))
                    normalized_ted = 2 * ted / (len(decomp1) + len(decomp2) + ted)
                    similarity = max(similarity, 1 - normalized_ted)
        else:
            for decomp1 in decomps_1:
                for decomp2 in decomps_2:
                    if not safe:
                        ed = edit_distance(decomp1, decomp2)
                    else:
                        ed = edit_distance(safe_encode_string(decomp1), safe_encode_string(decomp2))
                    normalized_ed = ed / max(len(decomp1), len(decomp2))
                    similarity = max(similarity, 1 - normalized_ed)

        return similarity

    def pronunciation_similarity(self, char1, char2):
        """
        >>> c = CharFuncs('data/char_meta.txt')
        >>> c.pronunciation_similarity('牛', '午')
        0.27999999999999997
        >>> c.pronunciation_similarity('由', '田')
        0.09

        """
        assert char1 in self.data
        assert char2 in self.data
        pronunciations1 = self.data[char1]["pronunciation"]
        pronunciations2 = self.data[char2]["pronunciation"]

        if pronunciations1[0] == 'null' or pronunciations2 == 'null':
            return 0.0
        else:

            pronunciations1 = pronunciations1.split(';')  # separate by lan
            pronunciations2 = pronunciations2.split(';')  # separate by lan

            similarity = 0.0
            count = 0
            for pron_lan1, pron_lan2 in zip(pronunciations1, pronunciations2):
                if (pron_lan1 == 'null') or (pron_lan2 == 'null'):
                    pass
                else:
                    similarity_lan = 0.0
                    for p1 in pron_lan1.split(','):
                        for p2 in pron_lan2.split(','):
                            tmp_sim = 1 - edit_distance(p1, p2) / max(len(p1), len(p2))
                            similarity_lan = max(similarity_lan, tmp_sim)
                    similarity += similarity_lan
                    count += 1

            return similarity / count if count else 0


def string_to_tree(string):
    """
    This function converts ids string to a string that can be used as a tree input to APTED.
    Any Error raised by this function implies that the input string is invalid.
    >>> string_to_tree('⿱⿱⿰丿㇏⿰丿㇏⿱⿰丿㇏⿰丿㇏')  # 炎
    '{⿱{⿱{⿰{丿}{㇏}}{⿰{丿}{㇏}}}{⿱{⿰{丿}{㇏}}{⿰{丿}{㇏}}}}'
    >>> string_to_tree('⿱⿰丿㇏⿱一⿱⿻一丨一')  # 全
    '{⿱{⿰{丿}{㇏}}{⿱{一}{⿱{⿻{一}{丨}}{一}}}}'
    >>> string_to_tree('⿱⿰丿㇏⿻⿱一⿱⿻一丨一丷') # 金
    '{⿱{⿰{丿}{㇏}}{⿻{⿱{一}{⿱{⿻{一}{丨}}{一}}}{丷}}}'
    >>> string_to_tree('⿻⿻⿻一丨一⿴⿱⿰丨𠃌一一') # 車
    '{⿻{⿻{⿻{一}{丨}}{一}}{⿴{⿱{⿰{丨}{𠃌}}{一}}{一}}}'
    >>> string_to_tree('⿻⿻⿻一丨⿰丿㇏⿴⿱⿰丨𠃌一一') # 東
    '{⿻{⿻{⿻{一}{丨}}{⿰{丿}{㇏}}}{⿴{⿱{⿰{丨}{𠃌}}{一}}{一}}}'
    >>> string_to_tree('丿') # 丿
    '{丿}'
    >>> string_to_tree('⿻') # ⿻
    '{⿻}'
    """
    if string[0] in IDCS and len(string) != 1:
        bracket_stack = []
        tree = []

        def add_brackets(num):
            if num == 2:
                bracket_stack.extend(['}', '{', '}'])
            else:
                bracket_stack.extend(['}', '{', '}', '{', '}'])
            tree.append('{')

        global_just_put = '{'

        for c in string:
            tree.append(c)
            if c in IDCS:
                assert global_just_put != '}'
                add_brackets(IDCS[c])
                global_just_put = '{'
            else:
                just_put = ''
                while just_put != '{' and bracket_stack:
                    just_put = bracket_stack.pop(-1)
                    tree.append(just_put)
                global_just_put = just_put

        res = ''.join(tree)
        assert res[-1] == '}'
    else:
        assert len(string) == 1 or string == 'null'
        res = string[0]

    return '{' + res + '}'


def pinyin_map(standard_pinyin):
    """
    >>> pinyin_map('xuě')
    'xue3'
    >>> pinyin_map('xue')
    'xue'
    >>> pinyin_map('lǜ')
    'lü4'
    >>> pinyin_map('fá')
    'fa2'
    """
    tone = ''
    pinyin = ''

    assert ' ' not in standard_pinyin
    for c in standard_pinyin:
        if c in PINYIN:
            pinyin += PINYIN[c][0]
            assert tone == ''
            tone = str(PINYIN[c][1])
        else:
            pinyin += c
    pinyin += tone
    return pinyin


def parse_args():
    usage = '\n1. You can compute character similarity by:\n' \
            'python char_sim.py 午 牛 年 千\n' \
            '\n' \
            '2. You can use ted in computing character similarity by:\n' \
            'python char_sim.py 午 牛 年 千 -t\n' \
            '\n'
    parser = argparse.ArgumentParser(
        description='A script to compute Chinese character (Kanji) similarity', usage=usage)

    parser.add_argument('multiargs', nargs='*', type=str, default=None,
                        help='Chinese characters in question')
    parser.add_argument('--ted', '-t', action="store_true", default=False,
                        help='True=to use tree edit distence (TED)'
                             'False=to use string edit distance')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    c = CharFuncs('data/char_meta.txt')
    if not args.ted:
        for i, c1 in enumerate(args.multiargs):
            for c2 in args.multiargs[i:]:
                if c1 != c2:
                    print(f'For character pair ({c1}, {c2}):')
                    print(f'    v-sim = {c.shape_similarity(c1, c2)}')
                    print(f'    p-sim = {c.pronunciation_similarity(c1, c2)}\n')
    else:
        for i, c1 in enumerate(args.multiargs):
            for c2 in args.multiargs[i:]:
                if c1 != c2:
                    print(f'For character pair ({c1}, {c2}):')
                    print(f'    v-sim = {c.shape_similarity(c1, c2, as_tree=True)}')
                    print(f'    p-sim = {c.pronunciation_similarity(c1, c2)}\n')