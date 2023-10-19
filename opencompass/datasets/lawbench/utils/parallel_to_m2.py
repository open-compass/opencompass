import os
from modules.annotator import Annotator
from modules.tokenizer import Tokenizer
import argparse
from collections import Counter
from tqdm import tqdm
import torch
from collections import defaultdict
from multiprocessing import Pool
from opencc import OpenCC
import timeout_decorator

os.environ["TOKENIZERS_PARALLELISM"] = "false"

annotator, sentence_to_tokenized = None, None
cc = OpenCC("t2s")

@timeout_decorator.timeout(10)
def annotate_with_time_out(line):
    """
    :param line:
    :return:
    """
    sent_list = line.split("\t")[1:]
    source = sent_list[0]
    if args.segmented:
        source = source.strip()
    else:
        source = "".join(source.strip().split())
    output_str = ""
    for idx, target in enumerate(sent_list[1:]):
        try:
            if args.segmented:
                target = target.strip()
            else:
                target = "".join(target.strip().split())
            if not args.no_simplified:
                target = cc.convert(target)
            source_tokenized, target_tokenized = sentence_to_tokenized[source], sentence_to_tokenized[target]
            out, cors = annotator(source_tokenized, target_tokenized, idx)
            if idx == 0:
                output_str += "".join(out[:-1])
            else:
                output_str += "".join(out[1:-1])
        except Exception:
            raise Exception
    return output_str


def annotate(line):
    """
    :param line:
    :return:
    """
    sent_list = line.split("\t")[1:]
    source = sent_list[0]
    if args.segmented:
        source = source.strip()
    else:
        source = "".join(source.strip().split())
    output_str = ""
    for idx, target in enumerate(sent_list[1:]):
        try:
            if args.segmented:
                target = target.strip()
            else:
                target = "".join(target.strip().split())
            if not args.no_simplified:
                target = cc.convert(target)
            source_tokenized, target_tokenized = sentence_to_tokenized[source], sentence_to_tokenized[target]
            out, cors = annotator(source_tokenized, target_tokenized, idx)
            if idx == 0:
                output_str += "".join(out[:-1])
            else:
                output_str += "".join(out[1:-1])
        except Exception:
            raise Exception
    return output_str





def firsttime_process(args):
    tokenizer = Tokenizer(args.granularity, args.device, args.segmented, args.bpe)
    global annotator, sentence_to_tokenized
    annotator = Annotator.create_default(args.granularity, args.multi_cheapest_strategy)
    lines = open(args.file, "r", encoding="utf-8").read().strip().split("\n")  # format: id src tgt1 tgt2...
    # error_types = []

    with open(args.output, "w", encoding="utf-8") as f:
        count = 0
        sentence_set = set()
        sentence_to_tokenized = {}
        for line in lines:
            sent_list = line.split("\t")[1:]
            for idx, sent in enumerate(sent_list):
                if args.segmented:
                    # print(sent)
                    sent = sent.strip()
                else:
                    sent = "".join(sent.split()).strip()
                if idx >= 1:
                    if not args.no_simplified:
                        sentence_set.add(cc.convert(sent))
                    else:
                        sentence_set.add(sent)
                else:
                    sentence_set.add(sent)
        batch = []
        for sent in tqdm(sentence_set):
            count += 1
            if sent:
                batch.append(sent)
            if count % args.batch_size == 0:
                results = tokenizer(batch)
                for s, r in zip(batch, results):
                    sentence_to_tokenized[s] = r  # Get tokenization map.
                batch = []
        if batch:
            results = tokenizer(batch)
            for s, r in zip(batch, results):
                sentence_to_tokenized[s] = r  # Get tokenization map.

        timeout_indices = []

        # 单进程模式
        for idx, line in enumerate(tqdm(lines)):
            try:
                ret = annotate_with_time_out(line)
            except Exception:
                timeout_indices.append(idx)
    return timeout_indices



def main(args):
    timeout_indices = firsttime_process(args)
    tokenizer = Tokenizer(args.granularity, args.device, args.segmented, args.bpe)
    global annotator, sentence_to_tokenized
    annotator = Annotator.create_default(args.granularity, args.multi_cheapest_strategy)
    lines = open(args.file, "r", encoding="utf-8").read().strip().split("\n")
    new_lines = []# format: id src tgt1 tgt2...

    with open(args.output, "w", encoding="utf-8") as f:
        count = 0
        sentence_set = set()
        sentence_to_tokenized = {}
        for line_idx, line in enumerate(lines):

            if line_idx in timeout_indices:
                # print(f"line before split: {line}")
                line_split = line.split("\t")
                line_number, sent_list = line_split[0], line_split[1:]
                assert len(sent_list) == 2
                sent_list[-1] = " 无"
                line = line_number + "\t" + "\t".join(sent_list)
                # print(f"line time out: {line}")
                new_lines.append(line)
            else:
                new_lines.append(line)

            sent_list = line.split("\t")[1:]
            for idx, sent in enumerate(sent_list):
                if args.segmented:
                    # print(sent)
                    sent = sent.strip()
                else:
                    sent = "".join(sent.split()).strip()
                if idx >= 1:
                    if not args.no_simplified:
                        sentence_set.add(cc.convert(sent))
                    else:
                        sentence_set.add(sent)
                else:
                    sentence_set.add(sent)
        batch = []
        for sent in tqdm(sentence_set):
            count += 1
            if sent:
                batch.append(sent)
            if count % args.batch_size == 0:
                results = tokenizer(batch)
                for s, r in zip(batch, results):
                    sentence_to_tokenized[s] = r  # Get tokenization map.
                batch = []
        if batch:
            results = tokenizer(batch)
            for s, r in zip(batch, results):
                sentence_to_tokenized[s] = r  # Get tokenization map.
    
        # 单进程模式
        lines = new_lines
        for idx, line in enumerate(tqdm(lines)):
            ret = annotate(line)
            f.write(ret)
            f.write("\n") 

        # 多进程模式：仅在Linux环境下测试，建议在linux服务器上使用
        # with Pool(args.worker_num) as pool:
        #     for ret in pool.imap(annotate, tqdm(lines), chunksize=8):
        #         if ret:
        #             f.write(ret)
        #             f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose input file to annotate")
    parser.add_argument("-f", "--file", type=str, required=True, help="Input parallel file")
    parser.add_argument("-o", "--output", type=str, help="Output file", required=True)
    parser.add_argument("-b", "--batch_size", type=int, help="The size of batch", default=128)
    parser.add_argument("-d", "--device", type=int, help="The ID of GPU", default=0)
    parser.add_argument("-w", "--worker_num", type=int, help="The number of workers", default=16)
    parser.add_argument("-g", "--granularity", type=str, help="Choose char-level or word-level evaluation", default="char")
    parser.add_argument("-m", "--merge", help="Whether merge continuous replacement/deletion/insertion", action="store_true")
    parser.add_argument("-s", "--multi_cheapest_strategy", type=str, choices=["first", "all"], default="all")
    parser.add_argument("--segmented", help="Whether tokens have been segmented", action="store_true")  # 支持提前token化，用空格隔开
    parser.add_argument("--no_simplified", help="Whether simplifying chinese", action="store_true")  # 将所有corrections转换为简体中文
    parser.add_argument("--bpe", help="Whether to use bpe", action="store_true")  # 支持 bpe 切分英文单词
    args = parser.parse_args()
    main(args)
