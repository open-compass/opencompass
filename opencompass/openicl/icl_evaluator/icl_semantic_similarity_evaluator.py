import numpy as np

from opencompass.registry import ICL_EVALUATORS
from opencompass.utils.text_postprocessors import general_postprocess

from .icl_base_evaluator import BaseEvaluator

DEFAULT_ST_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'


@ICL_EVALUATORS.register_module()
class SemanticSimilarityEvaluator(BaseEvaluator):
    """Embedding-based semantic similarity evaluator.

    Measures the cosine similarity between the embedding of a prediction
    and its reference(s), using a sentence-transformers encoder. Unlike
    literal-overlap metrics (exact match, ROUGE), it is robust to
    paraphrasing and synonymous wording, which makes it especially
    suitable for Chinese and free-form generation tasks.

    Args:
        model (str): Name or path of the sentence-transformers model.
            Defaults to a compact multilingual model that works well for
            Chinese. See https://www.sbert.net/docs/pretrained_models.html
        device (str, optional): Torch device for the encoder, e.g.
            ``'cuda:0'``. Defaults to the sentence-transformers auto
            selection.
        threshold (float, optional): If set, additionally report
            ``semantic_accuracy``: the share of samples whose similarity
            reaches this threshold. Range 0-1.
        batch_size (int): Encoding batch size. Defaults to 32.
        postprocess (bool): Whether to apply the shared
            ``general_postprocess`` to texts before encoding. Defaults to
            True.

    Return:
        dict: ``semantic_similarity`` (mean cosine similarity, 0-100);
        plus ``semantic_accuracy`` when ``threshold`` is set; plus
        per-sample ``details`` with the similarity of every sample.
    """

    def __init__(self,
                 model: str = DEFAULT_ST_MODEL,
                 device: str = None,
                 threshold: float = None,
                 batch_size: int = 32,
                 postprocess: bool = True) -> None:
        super().__init__()
        self.model_name = model
        self.device = device
        self.threshold = threshold
        self.batch_size = batch_size
        self.postprocess = postprocess
        self._encoder = None  # lazy: avoid loading the model at startup

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name,
                                                device=self.device)
        return self._encoder

    def _encode(self, texts):
        """Encode texts to normalized embeddings. Injectable for tests."""
        return self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    @staticmethod
    def _pairwise_cosine(pred_emb, ref_emb):
        # Embeddings are L2-normalized, so the cosine similarity is the
        # dot product.
        return float(np.dot(pred_emb, ref_emb))

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different '
                    'length'}
        if len(predictions) == 0:
            return {'error': 'empty predictions and references'}

        if self.postprocess:
            predictions = [general_postprocess(str(p)) for p in predictions]

        pred_emb = self._encode(predictions)
        similarities = []
        details = []
        for idx, (pred, ref) in enumerate(zip(predictions, references)):
            # A reference may be a single string or a list of acceptable
            # answers; the best (max) similarity wins.
            ref_list = ref if isinstance(ref, (list, tuple)) else [ref]
            ref_list = [str(r) for r in ref_list]
            ref_emb = self._encode(ref_list)
            similarity = max(
                self._pairwise_cosine(pred_emb[idx], r_emb)
                for r_emb in ref_emb)
            similarities.append(similarity)
            details.append({
                'pred': pred,
                'answer': ref,
                'similarity': similarity,
            })

        result = {
            'semantic_similarity':
            100 * float(np.mean(similarities)),
            'details':
            details,
        }
        if self.threshold is not None:
            hits = [s >= self.threshold for s in similarities]
            result['semantic_accuracy'] = 100 * float(np.mean(hits))
        return result
