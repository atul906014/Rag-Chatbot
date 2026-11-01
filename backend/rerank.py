
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL

_model = None


def get_reranker():
    global _model

    if _model is None:
        _model = CrossEncoder(
            RERANKER_MODEL,
            max_length=256,
            device="cpu"
        )

    return _model


def rerank(query: str, docs: list, top_k: int = 4) -> list:
    if not docs:
        return []

    model = get_reranker()

    pairs = [
        (query, doc.page_content[:3000])
        for doc in docs
    ]

    scores = model.predict(
        pairs,
        batch_size=2,
        show_progress_bar=False
    )

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: float(x[1]),
        reverse=True
    )

    return [
        (doc, float(score))
        for doc, score in ranked[:top_k]
    ]
