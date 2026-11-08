"""Base vectorial local con ChromaDB."""
from __future__ import annotations

from domain.evidence import Evidence


class ChromaKB:
    def __init__(self, collection_name: str = "proyecto2", path: str = "./chroma_db") -> None:
        import chromadb

        client = chromadb.PersistentClient(path=path)
        self._col = client.get_or_create_collection(collection_name)

    def index(self, documents: list[Evidence]) -> None:
        self._col.upsert(
            ids=[f"doc-{i}" for i in range(len(documents))],
            documents=[d.text for d in documents],
            metadatas=[{"source": d.source} for d in documents],
        )

    def retrieve(self, query: str, k: int = 3) -> list[Evidence]:
        res = self._col.query(query_texts=[query], n_results=k)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        return [Evidence(source=m.get("source", "kb"), text=t) for t, m in zip(docs, metas)]
