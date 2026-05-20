from .vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, n_results: int = 4):
        self._vs = vector_store
        self._n_results = n_results

    def retrieve(
        self,
        query: str,
        topic_filter: str | None = None,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """
        Retrieve relevant context for the query.
        Enriches the query with the last user turn for better follow-up handling.
        Returns a formatted context string ready to be injected into the prompt.
        """
        enriched_query = query
        if conversation_history:
            # Append last assistant response keywords to improve follow-up retrieval
            last_user_turns = [
                m["content"] for m in conversation_history[-4:] if m["role"] == "user"
            ]
            if last_user_turns:
                enriched_query = " ".join(last_user_turns[-2:]) + " " + query

        chunks = self._vs.query(
            enriched_query,
            n_results=self._n_results,
            topic_filter=topic_filter,
        )

        if not chunks:
            return ""

        lines = ["### Relevant Knowledge Base Context\n"]
        for i, chunk in enumerate(chunks, 1):
            lines.append(f"[Source {i}: {chunk['topic']} — {chunk['section']}]")
            lines.append(chunk["text"])
            lines.append("")

        return "\n".join(lines)
