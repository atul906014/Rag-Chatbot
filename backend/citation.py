"""Citation metadata extraction for retrieval results."""
from typing import Dict, List
from langchain_core.documents import Document


class CitationExtractor:
    """Extract citation metadata from retrieved documents."""

    @staticmethod
    def extract_citation(doc: Document) -> Dict[str, str]:
        """Extract citation info from a document's metadata."""
        meta = getattr(doc, "metadata", {}) or {}
        return {
            "paper_id": str(meta.get("paper_id", "")),
            "section": str(meta.get("section_path", "")),
            "page": str(meta.get("page_num", "")),
            "chunk_id": str(meta.get("chunk_id", "")),
            "node_type": str(meta.get("node_type", "")),
        }

    @staticmethod
    def format_citation(citation: Dict[str, str]) -> str:
        """Format citation as a readable string."""
        citation = citation or {}
        parts = []
        if citation.get("paper_id"):
            parts.append(f"Paper: {citation['paper_id']}")
        if citation.get("section"):
            parts.append(f"Section: {citation['section']}")
        if citation.get("page"):
            parts.append(f"Page: {citation['page']}")
        return " | ".join(parts) if parts else "Unknown source"

    @staticmethod
    def extract_all(docs: List[Document]) -> List[Dict[str, str]]:
        """Extract citations from multiple documents."""
        return [CitationExtractor.extract_citation(doc) for doc in docs if doc is not None]

    @staticmethod
    def extract_first(docs: List[Document]) -> Dict[str, str]:
        """Return the citation metadata for the first document, or an empty dict."""
        citations = CitationExtractor.extract_all(docs)
        return citations[0] if citations else {}
