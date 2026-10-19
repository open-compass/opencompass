from ltp import LTP
from typing import List
from pypinyin import pinyin, Style, lazy_pinyin
import torch
import os
import functools

class Tokenizer:
    """
    分词器
    """

    def __init__(self,
                 granularity: str = "word",
                 device: str = "cpu",
                 segmented: bool = False,
                 bpe: bool = False,
                 ) -> None:
        """
        构造函数
        :param mode: 分词模式，可选级别：字级别（char）、词级别（word）
        """
        self.ltp = None 
        if granularity == "word":
            self.ltp = LTP(device=torch.device(device) if torch.cuda.is_available() else torch.device("cpu"))
            self.ltp.add_words(words=["[缺失成分]"], max_window=6)
        self.segmented = segmented
        self.granularity = granularity
        if self.granularity == "word":
            self.tokenizer = self.split_word
        elif self.granularity == "char":
            self.tokenizer = functools.partial(self.split_char, bpe=bpe)
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        return "{:s}\nMode:{:s}\n}".format(str(self.__class__.__name__), self.mode)

    def __call__(self,
                 input_strings: List[str]
                 ) -> List:
        """
        分词函数
        :param input_strings: 需要分词的字符串列表
        :return: 分词后的结果列表，由元组组成，元组为(token,pos_tag,pinyin)的形式
        """
        if not self.segmented:
            input_strings = ["".join(s.split(" ")) for s in input_strings]
        results = self.tokenizer(input_strings)
        return results

    def split_char(self, input_strings: List[str], bpe=False) -> List:
        """
        分字函数
        :param input_strings: 需要分字的字符串
        :return: 分字结果
        """
        if bpe:
            from . import tokenization
            tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "data", "lawbench", "eval_assets", "chinese_vocab.txt"), do_lower_case=False)
        results = []
        for input_string in input_strings:
            if not self.segmented:  # 如果没有被分字，就按照每个字符隔开（不考虑英文标点的特殊处理，也不考虑BPE），否则遵循原分字结果
                segment_string = " ".join([char for char in input_string] if not bpe else tokenizer.tokenize(input_string))
            else:
                segment_string = input_string
                # print(segment_string)
            segment_string = segment_string.replace("[ 缺 失 成 分 ]", "[缺失成分]").split(" ")  # 缺失成分当成一个单独的token
            results.append([(char, "unk", pinyin(char, style=Style.NORMAL, heteronym=True)[0]) for char in segment_string])
        return results

    def split_word(self, input_strings: List[str]) -> List:
        """
        分词函数
        :param input_strings: 需要分词的字符串
        :return: 分词结果
        """
        if self.segmented:
            seg, hidden = self.ltp.seg([input_string.split(" ") for input_string in input_strings], is_preseged=True)
        else:
            seg, hidden = self.ltp.seg(input_strings)
        pos = self.ltp.pos(hidden)
        result = []
        for s, p in zip(seg, pos):
            pinyin = [lazy_pinyin(word) for word in s]
            result.append(list(zip(s, p, pinyin)))
        return result

if __name__ == "__main__":
    tokenizer = Tokenizer("word")
    print(tokenizer(["LAC是个优秀的分词工具", "百度是一家高科技公司"]))
