from typing import List, Tuple
from modules.alignment import read_cilin, read_confusion, Alignment
from modules.merger import Merger
from modules.classifier import Classifier

class Annotator:
    def __init__(self,
                 align: Alignment,
                 merger: Merger,
                 classifier: Classifier,
                 granularity: str = "word",
                 strategy: str = "first"):
        self.align = align
        self.merger = merger
        self.classifier = classifier
        self.granularity = granularity
        self.strategy = strategy

    @classmethod
    def create_default(cls, granularity: str = "word", strategy: str = "first"):
        """
        Default parameters used in the paper
        """
        semantic_dict, semantic_class = read_cilin()
        confusion_dict = read_confusion()
        align = Alignment(semantic_dict, confusion_dict, granularity)
        merger = Merger(granularity)
        classifier = Classifier(granularity)
        return cls(align, merger, classifier, granularity, strategy)

    def __call__(self,
                 src: List[Tuple],
                 tgt: List[Tuple],
                 annotator_id: int = 0,
                 verbose: bool = False):
        """
        Align sentences and annotate them with error type information
        """
        src_tokens = [x[0] for x in src]
        tgt_tokens = [x[0] for x in tgt]
        src_str = "".join(src_tokens)
        tgt_str = "".join(tgt_tokens)
        # convert to text form
        annotations_out = ["S " + " ".join(src_tokens) + "\n"]
        if tgt_str == "没有错误" or src_str == tgt_str:   # Error Free Case
            annotations_out.append(f"T{annotator_id} 没有错误\n")
            cors = [tgt_str]
            op, toks, inds = "noop", "-NONE-", (-1, -1)
            a_str = f"A {inds[0]} {inds[1]}|||{op}|||{toks}|||REQUIRED|||-NONE-|||{annotator_id}\n"
            annotations_out.append(a_str)
        elif tgt_str == "无法标注":  # Not Annotatable Case
            annotations_out.append(f"T{annotator_id} 无法标注\n")
            cors = [tgt_str]
            op, toks, inds = "NA", "-NONE-", (-1, -1)
            a_str = f"A {inds[0]} {inds[1]}|||{op}|||{toks}|||REQUIRED|||-NONE-|||{annotator_id}\n"
            annotations_out.append(a_str)
        else:  # Other
            align_objs = self.align(src, tgt)
            edit_objs = []
            align_idx = 0
            if self.strategy == "first":
                align_objs = align_objs[:1]
            for align_obj in align_objs:
                edits = self.merger(align_obj, src, tgt, verbose)
                if edits not in edit_objs:
                    edit_objs.append(edits)
                    annotations_out.append(f"T{annotator_id}-A{align_idx} " + " ".join(tgt_tokens) + "\n")
                    align_idx += 1
                    cors = self.classifier(src, tgt, edits, verbose)
                    # annotations_out = []
                    for cor in cors:
                        op, toks, inds = cor.op, cor.toks, cor.inds
                        a_str = f"A {inds[0]} {inds[1]}|||{op}|||{toks}|||REQUIRED|||-NONE-|||{annotator_id}\n"
                        annotations_out.append(a_str)
        annotations_out.append("\n")
        return annotations_out, cors
