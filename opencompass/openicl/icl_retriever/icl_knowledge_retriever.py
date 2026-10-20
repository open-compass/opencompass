"""Local Knowledge Retriever."""

from typing import List, Optional, Callable, Dict, Any

from opencompass.openicl.icl_retriever import BaseRetriever
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.registry import ICL_RETRIEVERS
from opencompass.utils import get_logger

logger = get_logger(__name__)

import os
import re
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, CSVLoader, UnstructuredFileLoader
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.faiss import dependable_faiss_import
from langchain.docstore.base import Docstore
from langchain.docstore.document import Document



VECTOR_SEARCH_SCORE_THRESHOLD = 500
CHUNK_SIZE = 50

class RetrievedFAISS(FAISS, VectorStore):
    def __init__(self,
                 embedding_function: Callable,
                 index: Any,
                 docstore: Docstore,
                 index_to_docstore_id: Dict[int, str],
                 normalize_L2: bool = False,
        ):
        super().__init__(embedding_function=embedding_function,
                         index=index,
                         docstore=docstore,
                         index_to_docstore_id=index_to_docstore_id,
                         normalize_L2=normalize_L2)
        self.score_threshold = VECTOR_SEARCH_SCORE_THRESHOLD
        self.chunk_size = CHUNK_SIZE
        self.chunk_conent = False

    def seperate_list(self, lines: List[int]) -> List[List[int]]:
        results = []
        cur_line = [lines[0]]
        docs_source = self.index_to_docstore_source(lines[0])
        for i in range(1, len(lines)):
            if lines[i - 1] + 1 == lines[i] and self.index_to_docstore_source(lines[i]) == docs_source:
                cur_line.append(lines[i])
            else:
                results.append(cur_line)
                cur_line = [lines[i]]
                docs_source = self.index_to_docstore_source(lines[i])
        results.append(cur_line)
        return results

    def similarity_search_with_score_by_vector(
            self, embedding: List[float], k: int = 4
            ) -> List[Document]:
        faiss = dependable_faiss_import()
        vector = np.array([embedding], dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, k)
        docs = []
        id_set = set()
        store_len = len(self.index_to_docstore_id)
        rearrange_id_list = False
        for j, i in enumerate(indices[0]):
            if i == -1 or 0 < self.score_threshold < scores[0][j]:
                continue
            if i in self.index_to_docstore_id:
                _id = self.index_to_docstore_id[i]
            else:
                continue
            doc = self.docstore.search(_id)
            if (not self.chunk_conent) or ("context_expand" in doc.metadata and not doc.metadata["context_expand"]):
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                doc.metadata["score"] = int(scores[0][j])
                docs.append(doc)
                continue

            id_set.add(i)
            docs_len = len(doc.page_content)
            for k in range(1, max(i, store_len - i)):
                break_flag = False
                if "context_expand_method" in doc.metadata and doc.metadata["context_expand_method"] == "forward":
                    expand_range = [i + k]
                elif "context_expand_method" in doc.metadata and doc.metadata["context_expand_method"] == "backward":
                    expand_range = [i - k]
                else:
                    expand_range = [i + k, i - k]
                for l in expand_range:
                    if l not in id_set and 0 <= l < len(self.index_to_docstore_id):
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        if docs_len + len(doc0.page_content) > self.chunk_size or doc0.metadata["source"] != \
                                doc.metadata["source"]:
                            break_flag = True
                            break
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
                            rearrange_id_list = True
                if break_flag:
                    break
        if (not self.chunk_conent) or (not rearrange_id_list):
            return docs
        if len(id_set) == 0 and self.score_threshold > 0:
            return []
        id_list = sorted(list(id_set))
        id_lists = self.seperate_list(id_list)
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    doc = deepcopy(self.docstore.search(_id))
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += " " + doc0.page_content
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
            doc.metadata["score"] = int(doc_score)
            docs.append(doc)
        return docs

    def list_docs(self):
        return list(v.metadata["source"] for v in self.docstore._dict.values())

    def index_to_docstore_source(self,i:int):
        _id = self.index_to_docstore_id[i]
        doc = self.docstore.search(_id)
        return doc.metadata["source"]

class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(
            self,
            max_length: int,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.max_length = max_length

    def split_text(
            self,
            text: str,
            is_pdf: bool = False,
            ) -> List[str]:
        if is_pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)

        text = re.sub(r'([;；.!?。！？\?])([^”’])', r"\1\n\2", text)
        text = re.sub(r'(\.{6})([^"’”」』])', r"\1\n\2", text)
        text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)
        text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
        text = text.rstrip()
        lines = [i for i in text.split("\n") if i]
        for cur_line in lines:
            if len(cur_line) > self.max_length:
                sub_lines1 = re.sub(r'([,，.]["’”」』]{0,2})([^,，.])', r'\1\n\2', cur_line).split("\n")
                for cur_s_line1 in sub_lines1:
                    if len(cur_s_line1) > self.max_length:
                        sub_lines2 = re.sub(r'([\n]{1,}| {2,}["’”」』]{0,2})([^\s])', r'\1\n\2', cur_s_line1).split("\n")
                        for cur_s_line2 in sub_lines2:
                            if len(cur_s_line2) > self.max_length:
                                cur_s_line3 = re.sub('( ["’”」』]{0,2})([^ ])', r'\1\n\2', cur_s_line2)
                                cur_s_idx2 = sub_lines2.index(cur_s_line2)
                                sub_lines2 = sub_lines2[:cur_s_idx2] + [i for i in cur_s_line3.split("\n") if i] + sub_lines2[cur_s_idx2 + 1:]
                        cur_s_idx1 = sub_lines1.index(cur_s_line1)
                        sub_lines1 = sub_lines1[:cur_s_idx1] + [i for i in sub_lines2 if i] + sub_lines1[cur_s_idx1 + 1:]

                cur_idx = lines.index(cur_line)
                lines = lines[:cur_idx] + [i for i in sub_lines1 if i] + lines[cur_idx + 1:]
        return lines

def load_knowledge(
        knowledge_doc: str,
        sentence_max_length: int
        ) -> List[Document]:
    """
    Load and split knowledge documents from .txt or .csv formats.

    knowledge_doc (`str`): Path to the knowledge document file.
    sentence_max_length (`str`): Maximum length of a sentence in terms of tokens.
    """
    text_splitter = ChineseTextSplitter(max_length=sentence_max_length)
    if knowledge_doc.lower().endswith(".txt"):
        loader = TextLoader(knowledge_doc, autodetect_encoding=True)
        docs = loader.load_and_split(text_splitter)
    elif knowledge_doc.lower().endswith(".csv"):
        loader = CSVLoader(knowledge_doc)
        docs = loader.load()
    else:
        loader = UnstructuredFileLoader(knowledge_doc, mode="elements")
        docs = loader.load_and_split(text_splitter=text_splitter)
    return docs

class LocalKnowledgeBase:
    """Local Knowledge Base.

    Args:
        embedding_path (`Optional[str]`): The path or name of the 
            pre-trained embedding model used for encoding text.
        topk (`int`): The number of most similar knowledge 
            documents to retrieve for a given query.
        knowledge_docs (`List`): Files containing the knowledge base,
            supporting txt, csv formats.
        sentence_max_length (`int`): Maximum length of a sentence
            in terms of tokens for processing.
        vector_store_path (`str or os.PathLike`): Path to save or load
            pre-computed document vectors.
        device (`Optional[str]`): The device (CPU or GPU) to
            run the embedding model on.
    """
    def __init__(
        self,
        embedding_path: str,
        topk: int,
        knowledge_docs: List[str],
        sentence_max_length: int,
        vector_store_path: str or os.PathLike = None,
        device: Optional[str] = None,
    ) -> None:
        from langchain.embeddings.huggingface import HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_path,
            model_kwargs={'device': device}
        )
        self.topk = topk

        docs = sum([load_knowledge(knowledge_doc=cur_doc, sentence_max_length=sentence_max_length) for cur_doc in knowledge_docs], [])

        if vector_store_path is None:
            vector_store_path = os.path.join(
                os.path.commonprefix(knowledge_docs).rsplit('/', 1)[0],
                "vector_store")

        if os.path.isdir(vector_store_path) and "index.faiss" in os.listdir(vector_store_path):
            logger.info(f'Loading from existing vector store ({vector_store_path})...')
            self.vector_store = RetrievedFAISS.load_local(vector_store_path, self.embeddings)
            self.vector_store.add_documents(docs)
        else:
            logger.info(f'Constructing vector store ({vector_store_path})...')
            self.vector_store = RetrievedFAISS.from_documents(docs, self.embeddings)

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.vector_store.save_local(vector_store_path)

        logger.info(f'Vector store is ready.')

    def retrieve_one(self, query: str, separator: str = ' ') -> str:
        """Retrieve the most relevant knowledge documents based on a query."""
        related_docs_with_score = self.vector_store.similarity_search_with_score(
            query,
            k=self.topk)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return separator.join([cur_doc.page_content for cur_doc in related_docs_with_score])

@ICL_RETRIEVERS.register_module()
class KnowledgeRetriever(BaseRetriever):
    """Local Knowledge Retriever. The retriever returns related local knowledge for all queries.

    Args:
        dataset (`BaseDataset`): Any BaseDataset instances.
            Attributes of ``reader``, ``train`` and ``test`` will be used.
        knowledge_docs (`List`): Files containing the knowledge base,
            supporting txt, csv formats.
        retrieve_keys (`List`): Keys of the test sample that require
            indexing of relevant knowledge.
        embedding_path (`Optional[str]`): The path or name of the 
            pre-trained embedding model used for encoding text.
        ice_eos_token (`Optional[str]`): The end of sentence token for
            in-context example template when origin `PromptTemplate` is
            provided. Defaults to ''.
    """
    def __init__(self,
                 dataset,
                 knowledge_docs: List,
                 retrieve_keys: List,
                 embedding_path: Optional[str] = 'GanymedeNil/text2vec-large-chinese',
                 ice_eos_token: Optional[str] = '') -> None:
        super().__init__(dataset, '', ice_eos_token, 0)
        self.knowledge_ds = None
        self.retrieve_keys = retrieve_keys

        self.local_knowledge_base = LocalKnowledgeBase(
            embedding_path=embedding_path,
            knowledge_docs=knowledge_docs,
            topk=3,
            sentence_max_length=100)

    def retrieve(self) -> List[List]:
        """Construct the knowledge base associated with test each sample and retrieve the sequential indices."""

        logger.info('Retrieving data for test set...')
        rtr_idx_list = [[i] for i in range(len(self.test_ds))]
        self.knowledge_ds = [
            {'knowledge': '; '.join([
                self.local_knowledge_base.retrieve_one(cur_d[option_key])
                for option_key in self.retrieve_keys
            ])} for cur_d in tqdm(self.test_ds)]
        return rtr_idx_list

    def generate_ice(self,
                     idx_list: List[int],
                     ice_template: Optional[PromptTemplate] = None) -> str:
        """Generate the knowledge-related example for one test example.

        Args:
            idx_list (`List[int]`): The index of knowledge-related examples for the
                test example.
            ice_template (`Optional[PromptTemplate]`): The template for
                knowledge-related example. Defaults to None.
        """
        assert self.knowledge_ds is not None, (
            'knowledge_ds must be set first in retrieve method')

        if ice_template is None:
            assert len(
                idx_list
            ) == 0, 'You have not specified ice_template while retrieving examples from train set! Please either specify ice_template or use `ZeroRetriever`.'  # noqa

        generated_ice_list = []
        for idx in idx_list:
            generated_ice_list.append(
                ice_template.generate_ice_item(
                    self.knowledge_ds[idx],
                    ''))

        generated_ice = self.ice_separator.join(
            generated_ice_list) + self.ice_eos_token
        return generated_ice
