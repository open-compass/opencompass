import numpy as np
from typing import List, Tuple, Dict
from modules.tokenizer import Tokenizer
import os
from string import punctuation

REAL_PATH = os.path.split(os.path.realpath(__file__))[0]
chinese_punct = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"
english_punct = punctuation
punct = chinese_punct + english_punct

def check_all_chinese(word):
    """
    判断一个单词是否全部由中文组成
    :param word:
    :return:
    """
    return all(['\u4e00' <= ch <= '\u9fff' for ch in word])

def read_cilin():
    """
    Cilin 詞林 is a thesaurus with semantic information
    """
    # TODO -- fix this path
    lines = open(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "data", "lawbench", "eval_assets", "cilin.txt"), "r", encoding="gbk").read().strip().split("\n")
    semantic_dict = {}
    semantic_classes = {}
    for line in lines:
        code, *words = line.split(" ")
        for word in words:
            semantic_dict[word] = code
        # make reverse dict
        if code in semantic_classes:
            semantic_classes[code] += words
        else:
            semantic_classes[code] = words
    return semantic_dict, semantic_classes


def read_confusion():
    confusion_dict = {}
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "data", "lawbench", "eval_assets", "confusion_dict.txt"), "r", encoding="utf-8") as f:
        for line in f:
            li = line.rstrip('\n').split(" ")
            confusion_dict[li[0]] = li[1:]
    return confusion_dict

class Alignment:
    """
    对齐错误句子和正确句子，
    使用编辑距离算法抽取编辑操作
    """

    def __init__(
            self,
            semantic_dict: Dict,
            confusion_dict: Dict,
            granularity: str = "word",
            ) -> None:
        """
        构造函数
        :param semantic_dict: 语义词典（大词林）
        :param confusion_dict: 字符混淆集
        """
        self.insertion_cost = 1
        self.deletion_cost = 1
        self.semantic_dict = semantic_dict
        self.confusion_dict = confusion_dict
        # Because we use character level tokenization, this doesn't currently use POS
        self._open_pos = {}  # 如果是词级别，还可以利用词性是否相同来计算cost
        self.granularity = granularity  # word-level or character-level
        self.align_seqs = []
        
    def __call__(self,
                 src: List[Tuple],
                 tgt: List[Tuple],
                 verbose: bool = False):
        cost_matrix, oper_matrix = self.align(src, tgt)
        align_seq = self.get_cheapest_align_seq(oper_matrix)

        if verbose:
            print("========== Seg. and POS: ==========")
            print(src)
            print(tgt)
            print("========== Cost Matrix ==========")
            print(cost_matrix)
            print("========== Oper Matrix ==========")
            print(oper_matrix)
            print("========== Alignment ==========")
            print(align_seq)
            print("========== Results ==========")
            for a in align_seq:
                print(a[0], src[a[1]: a[2]], tgt[a[3]: a[4]])
        return align_seq

    def _get_semantic_class(self, word):
        """
        NOTE: Based on the paper:
        Improved-Edit-Distance Kernel for Chinese Relation Extraction
        获取每个词语的语义类别（基于大词林，有三个级别）
        """
        if word in self.semantic_dict:
            code = self.semantic_dict[word]
            high, mid, low = code[0], code[1], code[2:4]
            return high, mid, low
        else:  # unknown
            return None

    @staticmethod
    def _get_class_diff(a_class, b_class):
        """
        d == 3 for equivalent semantics
        d == 0 for completely different semantics
        根据大词林的信息，计算两个词的语义类别的差距
        """
        d = sum([a == b for a, b in zip(a_class, b_class)])
        return d

    def _get_semantic_cost(self, a, b):
        """
        计算基于语义信息的替换操作cost
        :param a: 单词a的语义类别
        :param b: 单词b的语义类别
        :return: 替换编辑代价
        """
        a_class = self._get_semantic_class(a)
        b_class = self._get_semantic_class(b)
        # unknown class, default to 1
        if a_class is None or b_class is None:
            return 4
        elif a_class == b_class:
            return 0
        else:
            return 2 * (3 - self._get_class_diff(a_class, b_class))

    def _get_pos_cost(self, a_pos, b_pos):
        """
        计算基于词性信息的编辑距离cost
        :param a_pos: 单词a的词性
        :param b_pos: 单词b的词性
        :return: 替换编辑代价
        """
        if a_pos == b_pos:
            return 0
        elif a_pos in self._open_pos and b_pos in self._open_pos:
            return 0.25
        else:
            return 0.499

    def _get_char_cost(self, a, b, pinyin_a, pinyin_b):
        """
        NOTE: This is a replacement of ERRANTS lemma cost for Chinese
        计算基于字符相似度的编辑距离cost
        """
        if not (check_all_chinese(a) and check_all_chinese(b)):
            return 0.5
        if len(a) > len(b):
            a, b = b, a
            pinyin_a, pinyin_b = pinyin_b, pinyin_a
        if a == b:
            return 0
        else:
            return self._get_spell_cost(a, b, pinyin_a, pinyin_b)

    def _get_spell_cost(self, a, b, pinyin_a, pinyin_b):
        """
        计算两个单词拼写相似度，分别由字形相似度和字音相似度组成
        :param a: 单词a
        :param b: 单词b，且单词a的长度小于等于b
        :param pinyin_a: 单词a的拼音
        :param pinyin_b: 单词b的拼音
        :return: 替换操作cost
        """
        count = 0
        for i in range(len(a)):
            for j in range(len(b)):
                if a[i] == b[j] or (set(pinyin_a) & set(pinyin_b)) or (b[j] in self.confusion_dict.keys() and a[i] in self.confusion_dict[b[j]]) or (a[i] in self.confusion_dict.keys() and b[j] in self.confusion_dict[a[i]]):
                    count += 1
                    break
        return (len(a) - count) / (len(a) * 2)

    def get_sub_cost(self, a_seg, b_seg):
        """
        Calculate the substitution cost between words a and b
        计算两个单词替换操作的编辑cost，最大为2，等于一次删除和一次添加
        """
        if a_seg[0] == b_seg[0]:
            return 0

        if self.granularity == "word":  # 词级别可以额外利用词性信息
            semantic_cost = self._get_semantic_cost(a_seg[0], b_seg[0]) / 6.0
            pos_cost = self._get_pos_cost(a_seg[1], b_seg[1])
            char_cost = self._get_char_cost(a_seg[0], b_seg[0], a_seg[2], b_seg[2])
            return semantic_cost + pos_cost + char_cost
        else:  # 字级别只能利用字义信息（从大词林中获取）和字面相似度信息
            semantic_cost = self._get_semantic_cost(a_seg[0], b_seg[0]) / 6.0
            if a_seg[0] in punct and b_seg[0] in punct:
                pos_cost = 0.0
            elif a_seg[0] not in punct and b_seg[0] not in punct:
                pos_cost = 0.25
            else:
                pos_cost = 0.499
            # pos_cost = 0.0 if (a_seg[0] in punct and b_seg[0] in punct) or (a_seg[0] not in punct and b_seg[0] not in punct) else 0.5
            char_cost = self._get_char_cost(a_seg[0], b_seg[0], a_seg[2], b_seg[2])
            return semantic_cost + char_cost + pos_cost

    def align(self,
              src: List[Tuple],
              tgt: List[Tuple]):
        """
        Based on ERRANT's alignment
        基于改进的动态规划算法，为原句子的每个字打上编辑标签，以便使它能够成功转换为目标句子。
        编辑操作类别：
        1) M：Match，即KEEP，即当前字保持不变
        2) D：Delete，删除，即当前字需要被删除
        3) I：Insert，插入，即当前字需要被插入
        4) T：Transposition，移位操作，即涉及到词序问题
        """
        cost_matrix = np.zeros((len(src) + 1, len(tgt) + 1))  # 编辑cost矩阵
        oper_matrix = np.full(
            (len(src) + 1, len(tgt) + 1), "O", dtype=object
        )  # 操作矩阵
        # Fill in the edges
        for i in range(1, len(src) + 1):
            cost_matrix[i][0] = cost_matrix[i - 1][0] + 1
            oper_matrix[i][0] = ["D"]
        for j in range(1, len(tgt) + 1):
            cost_matrix[0][j] = cost_matrix[0][j - 1] + 1
            oper_matrix[0][j] = ["I"]

        # Loop through the cost matrix
        for i in range(len(src)):
            for j in range(len(tgt)):
                # Matches
                if src[i][0] == tgt[j][0]:  # 如果两个字相等，则匹配成功（Match），编辑距离为0
                    cost_matrix[i + 1][j + 1] = cost_matrix[i][j]
                    oper_matrix[i + 1][j + 1] = ["M"]
                # Non-matches
                else:
                    del_cost = cost_matrix[i][j + 1] + self.deletion_cost  # 由删除动作得到的总cost
                    ins_cost = cost_matrix[i + 1][j] + self.insertion_cost  # 由插入动作得到的总cost
                    sub_cost = cost_matrix[i][j] + self.get_sub_cost(
                        src[i], tgt[j]
                    )  # 由替换动作得到的总cost
                    # Calculate transposition cost
                    # 计算移位操作的总cost
                    trans_cost = float("inf")
                    k = 1
                    while (
                            i - k >= 0
                            and j - k >= 0
                            and cost_matrix[i - k + 1][j - k + 1]
                            != cost_matrix[i - k][j - k]
                    ):
                        p1 = sorted([a[0] for a in src][i - k: i + 1])
                        p2 = sorted([b[0] for b in tgt][j - k: j + 1])
                        if p1 == p2:
                            trans_cost = cost_matrix[i - k][j - k] + k
                            break
                        k += 1

                    costs = [trans_cost, sub_cost, ins_cost, del_cost]
                    ind = costs.index(min(costs))
                    cost_matrix[i + 1][j + 1] = costs[ind]
                    #     ind = costs.index(costs[ind], ind+1)
                    for idx, cost in enumerate(costs):
                        if cost == costs[ind]:
                            if idx == 0:
                                if oper_matrix[i + 1][j + 1] == "O":
                                    oper_matrix[i + 1][j + 1] = ["T" + str(k + 1)]
                                else:
                                    oper_matrix[i + 1][j + 1].append("T" + str(k + 1))
                            elif idx == 1:
                                if oper_matrix[i + 1][j + 1] == "O":
                                    oper_matrix[i + 1][j + 1] = ["S"]
                                else:
                                    oper_matrix[i + 1][j + 1].append("S")
                            elif idx == 2:
                                if oper_matrix[i + 1][j + 1] == "O":
                                    oper_matrix[i + 1][j + 1] = ["I"]
                                else:
                                    oper_matrix[i + 1][j + 1].append("I")
                            else:
                                if oper_matrix[i + 1][j + 1] == "O":
                                    oper_matrix[i + 1][j + 1] = ["D"]
                                else:
                                    oper_matrix[i + 1][j + 1].append("D")
        return cost_matrix, oper_matrix

    def _dfs(self, i, j, align_seq_now, oper_matrix, strategy="all"):
        """
        深度优先遍历，获取最小编辑距离相同的所有序列
        """
        if i + j == 0:
            self.align_seqs.append(align_seq_now)
        else:
            ops = oper_matrix[i][j]  # 可以类比成搜索一棵树从根结点到叶子结点的所有路径
            if strategy != "all": ops = ops[:1]
            for op in ops:
                if op in {"M", "S"}:
                    self._dfs(i - 1, j - 1, align_seq_now + [(op, i - 1, i, j - 1, j)], oper_matrix, strategy)
                elif op == "D":
                    self._dfs(i - 1, j, align_seq_now + [(op, i - 1, i, j, j)], oper_matrix, strategy)
                elif op == "I":
                    self._dfs(i, j - 1, align_seq_now + [(op, i, i, j - 1, j)], oper_matrix, strategy)
                else:
                    k = int(op[1:])
                    self._dfs(i - k, j - k, align_seq_now + [(op, i - k, i, j - k, j)], oper_matrix, strategy)

    def get_cheapest_align_seq(self, oper_matrix):
        """
        回溯获得编辑距离最小的编辑序列
        """
        self.align_seqs = []
        i = oper_matrix.shape[0] - 1
        j = oper_matrix.shape[1] - 1
        if abs(i - j) > 10:
            self._dfs(i, j , [], oper_matrix, "first")
        else:
            self._dfs(i, j , [], oper_matrix, "all")
        final_align_seqs = [seq[::-1] for seq in self.align_seqs]
        return final_align_seqs


if __name__ == "__main__":
    tokenizer = Tokenizer("word")
    semantic_dict, semantic_class = read_cilin()
    confusion_dict = read_confusion()
    alignment = Alignment(semantic_dict, confusion_dict)
    sents = ["首先 ， 我们 得 准备 : 大 虾六 到 九 只 、 盐 一 茶匙 、 已 搾 好 的 柠檬汁 三 汤匙 、 泰国 柠檬 叶三叶 、 柠檬 香草 一 根 、 鱼酱 两 汤匙 、 辣椒 6 粒 ， 纯净 水 4量杯 、 香菜 半量杯 和 草菇 10 个 。".replace(" ", ""), "首先 ， 我们 得 准备 : 大 虾六 到 九 只 、 盐 一 茶匙 、 已 榨 好 的 柠檬汁 三 汤匙 、 泰国 柠檬 叶三叶 、 柠檬 香草 一 根 、 鱼酱 两 汤匙 、 辣椒 六 粒 ， 纯净 水 四 量杯 、 香菜 半量杯 和 草菇 十 个 。".replace(" ", "")]
    src, tgt = tokenizer(sents)
    alignment(src, tgt, verbose=True)