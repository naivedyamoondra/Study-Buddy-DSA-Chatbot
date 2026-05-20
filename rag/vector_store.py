import chromadb
from chromadb import EmbeddingFunction
from sentence_transformers import SentenceTransformer


class _SentenceTransformerEF(EmbeddingFunction):
    """Wraps sentence-transformers so ChromaDB can call it."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def __call__(self, input):  # input is list[str]
        return self._model.encode(input, convert_to_numpy=True).tolist()


class VectorStore:
    COLLECTION_NAME = "dsa_knowledge"

    def __init__(self, persist_dir: str = "./chroma_db"):
        self._ef = _SentenceTransformerEF()
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def is_populated(self) -> bool:
        return self._collection.count() > 0

    def add_documents(self, documents: list[dict]) -> None:
        """Add a list of {"id", "text", "topic", "section"} dicts."""
        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [{"topic": doc["topic"], "section": doc["section"]} for doc in documents]
        self._collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def query(self, query_text: str, n_results: int = 5, topic_filter: str | None = None) -> list[dict]:
        """Return top-n relevant chunks for the query."""
        where = {"topic": topic_filter} if topic_filter else None
        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self._collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "topic": meta["topic"],
                "section": meta["section"],
                "score": round(1 - dist, 3),  # cosine similarity
            })
        return chunks
