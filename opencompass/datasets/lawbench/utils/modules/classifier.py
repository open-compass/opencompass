from char_smi import CharFuncs
from collections import namedtuple
from pypinyin import pinyin, Style
import os
Correction = namedtuple(
    "Correction",
    [
        "op",
        "toks",
        "inds",
    ],
) 
char_smi = CharFuncs(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "data", "lawbench", "eval_assets", "char_meta.txt"))

def check_spell_error(src_span: str,
                      tgt_span: str,
                      threshold: float = 0.8) -> bool:
    if len(src_span) != len(tgt_span):
        return False
    src_chars = [ch for ch in src_span]
    tgt_chars = [ch for ch in tgt_span]
    if sorted(src_chars) == sorted(tgt_chars):  # 词内部字符异位
        return True
    for src_char, tgt_char in zip(src_chars, tgt_chars):
        if src_char != tgt_char:
            if src_char not in char_smi.data or tgt_char not in char_smi.data:
                return False
            v_sim = char_smi.shape_similarity(src_char, tgt_char)
            p_sim = char_smi.pronunciation_similarity(src_char, tgt_char)
            if v_sim + p_sim < threshold and not (
                    set(pinyin(src_char, style=Style.NORMAL, heteronym=True)[0]) & set(pinyin(tgt_char, style=Style.NORMAL, heteronym=True)[0])):
                return False
    return True

class Classifier:
    """
    错误类型分类器
    """
    def __init__(self,
                 granularity: str = "word"):

        self.granularity = granularity

    @staticmethod
    def get_pos_type(pos):
        if pos in {"n", "nd"}:
            return "NOUN"
        if pos in {"nh", "ni", "nl", "ns", "nt", "nz"}:
            return "NOUN-NE"
        if pos in {"v"}:
            return "VERB"
        if pos in {"a", "b"}:
            return "ADJ"
        if pos in {"c"}:
            return "CONJ"
        if pos in {"r"}:
            return "PRON"
        if pos in {"d"}:
            return "ADV"
        if pos in {"u"}:
            return "AUX"
        # if pos in {"k"}:  # TODO 后缀词比例太少，暂且分入其它
        #     return "SUFFIX"
        if pos in {"m"}:
            return "NUM"
        if pos in {"p"}:
            return "PREP"
        if pos in {"q"}:
            return "QUAN"
        if pos in {"wp"}:
            return "PUNCT"
        return "OTHER"

    def __call__(self,
                 src,
                 tgt,
                 edits,
                 verbose: bool = False):
        """
        为编辑操作划分错误类型
        :param src: 错误句子信息
        :param tgt: 正确句子信息
        :param edits: 编辑操作
        :param verbose: 是否打印信息
        :return: 划分完错误类型后的编辑操作
        """
        results = []
        src_tokens = [x[0] for x in src]
        tgt_tokens = [x[0] for x in tgt]
        for edit in edits:
            error_type = edit[0]
            src_span = " ".join(src_tokens[edit[1]: edit[2]])
            tgt_span = " ".join(tgt_tokens[edit[3]: edit[4]])
            # print(tgt_span)
            cor = None
            if error_type[0] == "T":
                cor = Correction("W", tgt_span, (edit[1], edit[2]))
            elif error_type[0] == "D":
                if self.granularity == "word":  # 词级别可以细分错误类型
                    if edit[2] - edit[1] > 1:  # 词组冗余暂时分为OTHER
                        cor = Correction("R:OTHER", "-NONE-", (edit[1], edit[2]))
                    else:
                        pos = self.get_pos_type(src[edit[1]][1])
                        pos = "NOUN" if pos == "NOUN-NE" else pos
                        pos = "MC" if tgt_span == "[缺失成分]" else pos
                        cor = Correction("R:{:s}".format(pos), "-NONE-", (edit[1], edit[2]))
                else:  # 字级别可以只需要根据操作划分类型即可
                    cor = Correction("R", "-NONE-", (edit[1], edit[2]))
            elif error_type[0] == "I":
                if self.granularity == "word":  # 词级别可以细分错误类型
                    if edit[4] - edit[3] > 1:  # 词组丢失暂时分为OTHER
                        cor = Correction("M:OTHER", tgt_span, (edit[1], edit[2]))
                    else:
                        pos = self.get_pos_type(tgt[edit[3]][1])
                        pos = "NOUN" if pos == "NOUN-NE" else pos
                        pos = "MC" if tgt_span == "[缺失成分]" else pos
                        cor = Correction("M:{:s}".format(pos), tgt_span, (edit[1], edit[2]))
                else:  # 字级别可以只需要根据操作划分类型即可
                    cor = Correction("M", tgt_span, (edit[1], edit[2]))
            elif error_type[0] == "S":
                if self.granularity == "word":  # 词级别可以细分错误类型
                    if check_spell_error(src_span.replace(" ", ""), tgt_span.replace(" ", "")):
                        cor = Correction("S:SPELL", tgt_span, (edit[1], edit[2]))
                        # Todo 暂且不单独区分命名实体拼写错误
                        # if edit[4] - edit[3] > 1:
                        #     cor = Correction("S:SPELL:COMMON", tgt_span, (edit[1], edit[2]))
                        # else:
                        #     pos = self.get_pos_type(tgt[edit[3]][1])
                        #     if pos == "NOUN-NE":  # 命名实体拼写有误
                        #         cor = Correction("S:SPELL:NE", tgt_span, (edit[1], edit[2]))
                        #     else:  # 普通词语拼写有误
                        #         cor = Correction("S:SPELL:COMMON", tgt_span, (edit[1], edit[2]))
                    else:
                        if edit[4] - edit[3] > 1:  # 词组被替换暂时分为OTHER
                            cor = Correction("S:OTHER", tgt_span, (edit[1], edit[2]))
                        else:
                            pos = self.get_pos_type(tgt[edit[3]][1])
                            pos = "NOUN" if pos == "NOUN-NE" else pos
                            pos = "MC" if tgt_span == "[缺失成分]" else pos
                            cor = Correction("S:{:s}".format(pos), tgt_span, (edit[1], edit[2]))
                else:  # 字级别可以只需要根据操作划分类型即可
                    cor = Correction("S", tgt_span, (edit[1], edit[2]))
            results.append(cor)
        if verbose:
            print("========== Corrections ==========")
            for cor in results:
                print("Type: {:s}, Position: {:d} -> {:d}, Target: {:s}".format(cor.op, cor.inds[0], cor.inds[1], cor.toks))
        return results

# print(pinyin("朝", style=Style.NORMAL))
