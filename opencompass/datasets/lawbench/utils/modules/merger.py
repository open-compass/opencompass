from itertools import groupby
from string import punctuation
from typing import List
from modules.tokenizer import Tokenizer
from modules.alignment import Alignment, read_cilin, read_confusion
import Levenshtein

class Merger:
    """
    合并编辑操作，从Token-Level转换为Span-Level
    """

    def __init__(self,
                 granularity: str = "word",
                 merge: bool = False):
        chinese_punct = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟–—‘'‛“”„‟…‧."
        self.punctuation = punctuation + chinese_punct
        self.not_merge_token = [punct for punct in self.punctuation]
        self.granularity = granularity
        self.merge = merge

    @staticmethod
    def _merge_edits(seq, tag="X"):
        if seq:
            return [(tag, seq[0][1], seq[-1][2], seq[0][3], seq[-1][4])]
        else:
            return seq

    @staticmethod
    def _check_revolve(span_a, span_b):
        span_a = span_a + span_a
        return span_b in span_a

    def _process_seq(self, seq, src_tokens, tgt_tokens):
        if len(seq) <= 1:
            return seq

        ops = [op[0] for op in seq]
        if set(ops) == {"D"} or set(ops) == {"I"}:
            return self._merge_edits(seq, set(ops).pop())

        if set(ops) == {"D", "I"} or set(ops) == {"I", "D"}:
            # do not merge this pattern_from_qua.txt
            return seq

        if set(ops) == {"S"}:
            if self.granularity == "word":
                return seq
            else:
                return self._merge_edits(seq, "S")

        if set(ops) == {"M"}:
            return self._merge_edits(seq, "M")

        return self._merge_edits(seq, "S")

    def __call__(self,
                 align_obj,
                 src: List,
                 tgt: List,
                 verbose: bool = False):
        """
        Based on ERRANT's merge, adapted for Chinese
        """
        src_tokens = [x[0] for x in src]
        tgt_tokens = [x[0] for x in tgt]
        edits = []
        # Split alignment into groups of M, T and rest. (T has a number after it)
        # Todo 一旦插入、删除、替换的对象中含有标点，那么不与其它编辑合并
        # Todo 缺失成分标签也不与其它编辑合并
        for op, group in groupby(
            align_obj,
            lambda x: x[0][0] if x[0][0] in {"M", "T"}  else False,
        ):
            group = list(group)
            # T is always split TODO: Evaluate this
            if op == "T":
                for seq in group:
                    edits.append(seq)
            # Process D, I and S subsequence
            else:
                # Turn the processed sequence into edits
                processed = self._process_seq(group, src_tokens, tgt_tokens)
                for seq in processed:
                    edits.append(seq)

        filtered_edits = []
        i = 0
        while i < len(edits):
            e1 = edits[i][0][0]

            if i < len(edits) - 2:
                e2 = edits[i + 1][0][0]
                e3 = edits[i + 2][0][0]

                # Find "S M S" patterns
                # Ex:
                #   S     M     S
                # 冬阴功  对  外国人
                # 外国人  对  冬阴功
                if e1 == "S" and e2 == "M" and e3 == "S":
                    w1 = "".join(src_tokens[edits[i][1]: edits[i][2]])
                    w2 = "".join(tgt_tokens[edits[i][3]: edits[i][4]])
                    w3 = "".join(src_tokens[edits[i + 2][1]: edits[i + 2][2]])
                    w4 = "".join(tgt_tokens[edits[i + 2][3]: edits[i + 2][4]])
                    if min([len(w1), len(w2), len(w3), len(w4)]) == 1:
                        if w1 == w4 and w2 == w3:
                            group = [edits[i], edits[i + 1], edits[i + 2]]
                            processed = self._merge_edits(group, "T" + str(edits[i+2][2] - edits[i][1]))
                            for seq in processed:
                                filtered_edits.append(seq)
                            i += 3
                        else:
                            filtered_edits.append(edits[i])
                            i += 1
                    else:
                        if Levenshtein.distance(w1, w4) <= 1 and Levenshtein.distance(w2, w3) <= 1:
                            group = [edits[i], edits[i + 1], edits[i + 2]]
                            processed = self._merge_edits(group, "T" + str(edits[i + 2][2] - edits[i][1]))
                            for seq in processed:
                                filtered_edits.append(seq)
                            i += 3
                        else:
                            filtered_edits.append(edits[i])
                            i += 1
                # Find "D M I" or "I M D" patterns
                # Ex:
                #   D        M              I
                # 旅游 去   陌生 的   地方
                #      去   陌生 的   地方  旅游
                elif (e1 == "D" and (e2 == "M" or e2.startswith("T")) and e3 == "I") or (e1 == "I" and (e2 == "M" or e2.startswith("T")) and e3 == "D"):
                    if e1 == "D":
                        delete_token = src_tokens[edits[i][1]: edits[i][2]]
                        insert_token = tgt_tokens[edits[i + 2][3]: edits[i + 2][4]]
                    else:
                        delete_token = src_tokens[edits[i + 2][1]: edits[i + 2][2]]
                        insert_token = tgt_tokens[edits[i][3]: edits[i][4]]
                    a, b = "".join(delete_token), "".join(insert_token)
                    if len(a) < len(b):
                        a, b = b, a
                    if a not in self.punctuation and b not in self.punctuation and len(a) - len(b) <= 1:
                        if len(b) == 1:
                            if a == b:
                                group = [edits[i], edits[i + 1], edits[i + 2]]
                                processed = self._merge_edits(group, "T" + str(edits[i+2][2] - edits[i][1]))
                                for seq in processed:
                                    filtered_edits.append(seq)
                                i += 3
                            else:
                                filtered_edits.append(edits[i])
                                i += 1
                        else:
                            if Levenshtein.distance(a, b) <= 1 or (len(a) == len(b) and self._check_revolve(a, b)):
                                group = [edits[i], edits[i + 1], edits[i + 2]]
                                processed = self._merge_edits(group, "T" + str(edits[i + 2][2] - edits[i][1]))
                                for seq in processed:
                                    filtered_edits.append(seq)
                                i += 3
                            else:
                                filtered_edits.append(edits[i])
                                i += 1
                    else:
                        filtered_edits.append(edits[i])
                        i += 1
                else:
                    if e1 != "M":
                        filtered_edits.append(edits[i])
                    i += 1
            else:
                if e1 != "M":
                    filtered_edits.append(edits[i])
                i += 1
        # In rare cases with word-level tokenization, the following error can occur:
        # M     D   S       M
        # 有    時  住      上層
        # 有        時住    上層
        # Which results in S: 時住 --> 時住
        # We need to filter this case out
        second_filter = []
        for edit in filtered_edits:  # 避免因为分词错误导致的mismatch现象
            span1 = "".join(src_tokens[edit[1] : edit[2]])
            span2 = "".join(tgt_tokens[edit[3] : edit[4]])

            if span1 != span2:
                if edit[0] == "S":
                    b = True
                    # In rare cases with word-level tokenization, the following error can occur:
                    # S       I     I       M
                    # 负责任               老师
                    # 负     责任   的     老师
                    # Which results in S: 负责任 --> 负 责任 的
                    # We need to convert this edit to I: --> 的

                    # 首部有重叠
                    common_str = ""
                    tmp_new_start_1 = edit[1]
                    for i in range(edit[1], edit[2]):
                        if not span2.startswith(common_str + src_tokens[i]):
                            break
                        common_str += src_tokens[i]
                        tmp_new_start_1 = i + 1
                    new_start_1, new_start_2 = edit[1], edit[3]
                    if common_str:
                        tmp_str = ""
                        for i in range(edit[3], edit[4]):
                            tmp_str += tgt_tokens[i]
                            if tmp_str == common_str:
                                new_start_1, new_start_2 = tmp_new_start_1, i + 1
                                # second_filter.append(("S", new_start_1, edit[2], i + 1, edit[4]))
                                b = False
                                break
                            elif len(tmp_str) > len(common_str):
                                break
                    # 尾部有重叠
                    common_str = ""
                    new_end_1, new_end_2 = edit[2], edit[4]
                    tmp_new_end_1 = edit[2]
                    for i in reversed(range(new_start_1, edit[2])):
                        if not span2.endswith(src_tokens[i] + common_str):
                            break
                        common_str = src_tokens[i] + common_str
                        tmp_new_end_1 = i
                    if common_str:
                        tmp_str = ""
                        for i in reversed(range(new_start_2, edit[4])):
                            tmp_str = tgt_tokens[i] + tmp_str
                            if tmp_str == common_str:
                                new_end_1, new_end_2 = tmp_new_end_1, i
                                b = False
                                break
                            elif len(tmp_str) > len(common_str):
                                break
                    if b:
                        second_filter.append(edit)
                    else:
                        if new_start_1 == new_end_1:
                            new_edit = ("I", new_start_1, new_end_1, new_start_2, new_end_2)
                        elif new_start_2 == new_end_2:
                            new_edit = ("D", new_start_1, new_end_1, new_start_2, new_end_2)
                        else:
                            new_edit = ("S", new_start_1, new_end_1, new_start_2, new_end_2)
                        second_filter.append(new_edit)
                else:
                    second_filter.append(edit)
        if verbose:
            print("========== Parallels ==========")
            print("".join(src_tokens))
            print("".join(tgt_tokens))
            print("========== Results ==========")
            for edit in second_filter:
                op = edit[0]
                s = " ".join(src_tokens[edit[1]: edit[2]])
                t = " ".join(tgt_tokens[edit[3]: edit[4]])
                print(f"{op}:\t{s}\t-->\t{t}")
            print("========== Infos ==========")
            print(str(src))
            print(str(tgt))
        return second_filter

if __name__ == "__main__":
    tokenizer = Tokenizer("char")
    semantic_dict, semantic_class = read_cilin()
    confusion_dict = read_confusion()
    alignment = Alignment(semantic_dict, confusion_dict)
    sents = [
        "所 以 印 度 对 全 世 界 人 没 有 说 服 不 要 吃 牛 肉 。".replace(
            " ", ""),
        "所 以 印 度 没 有 说 服 全 世 界 人 不 要 吃 牛 肉 。".replace(
            " ", "")]
    src, tgt = tokenizer(sents)
    align_obj = alignment(src, tgt)
    m = Merger()
    m(align_obj, src, tgt, verbose=True)