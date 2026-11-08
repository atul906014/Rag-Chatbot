from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL


reranker: CrossEncoder | None = None


def initialize_reranker() -> None:
    global reranker
    if reranker is None:
        reranker = CrossEncoder(
            RERANKER_MODEL,
            max_length=256,
            device="cpu",
        )

def rerank(query: str, docs: list, top_k: int = 4) -> list:
    if not docs:
        return []
    if reranker is None:
        raise RuntimeError("The reranker has not been initialized.")

    pairs = [
        (query, doc.page_content[:3000])
        for doc in docs
    ]

    scores = reranker.predict(
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
