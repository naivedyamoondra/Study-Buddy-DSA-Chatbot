import streamlit as st
from pathlib import Path

from rag.loader import load_documents
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from chatbot.client import StudyBuddyClient

KNOWLEDGE_BASE_DIR = str(Path(__file__).parent / "knowledge_base")
CHROMA_DIR = str(Path(__file__).parent / "chroma_db")

DSA_TOPICS = [
    "All Topics",
    "Arrays Strings",
    "Binary Search",
    "Linked Lists",
    "Stacks Queues",
    "Trees",
    "Graphs",
    "Sorting",
    "Dynamic Programming",
    "Hashing",
    "Recursion Backtracking",
]

DEPTH_OPTIONS = ["Beginner", "Intermediate", "Advanced"]

OLLAMA_MODELS = [
    "llama3.2",    # 2 GB — fast, good quality (recommended)
    "llama3.1",    # 4.7 GB — better reasoning
    "mistral",     # 4.1 GB — strong at coding
    "phi3",        # 2.2 GB — very fast, lighter
    "gemma2",      # 5.4 GB — good instruction following
]


# ── Cached resource initialisation ──────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base & building vector store...")
def init_vector_store() -> VectorStore:
    vs = VectorStore(persist_dir=CHROMA_DIR)
    if not vs.is_populated():
        documents = load_documents(KNOWLEDGE_BASE_DIR)
        vs.add_documents(documents)
    return vs


@st.cache_resource
def init_retriever(_vs: VectorStore) -> Retriever:
    return Retriever(_vs, n_results=4)


@st.cache_resource
def init_client() -> StudyBuddyClient:
    return StudyBuddyClient()


# ── Page setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Study Buddy — DSA Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Study Buddy")
st.caption("Local AI-powered DSA tutor — runs 100% on your machine, no internet needed")


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    model = st.selectbox(
        "Ollama Model",
        OLLAMA_MODELS,
        index=0,
        help="Make sure you've run: ollama pull <model-name>",
    )

    depth = st.selectbox(
        "Explanation Depth",
        DEPTH_OPTIONS,
        index=1,
        help="Beginner: simple analogies | Intermediate: terminology + code | Advanced: trade-offs + theory",
    )

    topic_choice = st.selectbox(
        "Focus Topic (optional)",
        DSA_TOPICS,
        help="Restrict retrieval to a specific DSA topic, or search across all.",
    )

    st.divider()

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "- Your question is embedded locally (sentence-transformers)\n"
        "- Relevant DSA notes are retrieved from ChromaDB (RAG)\n"
        "- Ollama generates the answer using retrieved context + chat history\n"
        "- Everything runs on your machine — zero API costs"
    )

    with st.expander("First time setup"):
        st.markdown("Download from **ollama.com/download**, then:")
        st.code("ollama serve", language="bash")
        st.code(f"ollama pull {model}", language="bash")

    with st.expander("Knowledge base topics"):
        for t in DSA_TOPICS[1:]:
            st.markdown(f"- {t}")


# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Load resources ────────────────────────────────────────────────────────────

vector_store = init_vector_store()
retriever = init_retriever(vector_store)
client = init_client()


# ── Render chat history ───────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask anything about DSA…"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    history = st.session_state.messages[:-1]
    topic_filter = None if topic_choice == "All Topics" else topic_choice
    context = retriever.retrieve(
        query=prompt,
        topic_filter=topic_filter,
        conversation_history=history,
    )

    with st.chat_message("assistant"):
        try:
            response = st.write_stream(
                client.chat_stream(
                    user_message=prompt,
                    retrieved_context=context,
                    conversation_history=history,
                    depth=depth,
                    model=model,
                )
            )
        except Exception as e:
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                st.error(
                    "Cannot connect to Ollama. Make sure it's running:\n\n"
                    "```\nollama serve\n```"
                )
            else:
                st.error(f"Error: {e}")
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": response})

    if context:
        with st.expander("Retrieved sources", expanded=False):
            st.markdown(context)
