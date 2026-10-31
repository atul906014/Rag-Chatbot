from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL

#Loaded once at import time and shared by all sessions.
# max_length caps tokens per (query, chunk) pair; 512 is the model's limit.
_model = CrossEncoder(RERANKER_MODEL, max_length=512)


def rerank(query: str, docs: list, top_k: int = 4) -> list[tuple]:
    """
    Score each (query, chunk) pair with the cross-encoder and return the
    top_k as [(doc, score), ...], best first.

    Scores are raw logits: clearly relevant chunks are typically positive and
    irrelevant ones strongly negative, which is what topic rejection relies on.
    """
    if not docs:
        return []

    pairs = [(query, d.page_content) for d in docs]
    scores = _model.predict(pairs, batch_size=16, show_progress_bar=False)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [(doc, float(score)) for doc, score in ranked[:top_k]]