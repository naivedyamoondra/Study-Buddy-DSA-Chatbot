# Study Buddy — DSA Chatbot

A local, fully offline RAG-based chatbot that tutors you on Data Structures and Algorithms. Powered by Ollama (free, runs on your machine), ChromaDB, and Streamlit.

---

## Features

- **RAG (Retrieval-Augmented Generation)** — answers are grounded in a curated DSA knowledge base, not just LLM memory
- **Conversation memory** — remembers the last 10 turns so follow-up questions work naturally
- **Adjustable explanation depth** — Beginner, Intermediate, or Advanced mode changes how the LLM explains concepts
- **Topic filter** — restrict retrieval to a specific DSA topic (e.g. only Graphs)
- **100% local** — no API keys, no internet required after setup, zero cost

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | [Ollama](https://ollama.com) (llama3.2, mistral, phi3, etc.) |
| Vector DB | [ChromaDB](https://www.trychroma.com) (local persistent store) |
| Embeddings | [sentence-transformers](https://www.sbert.net) (`all-MiniLM-L6-v2`) |
| UI | [Streamlit](https://streamlit.io) |

---

## How the RAG Pipeline Works

```
Your Question
     │
     ▼
┌─────────────────────────────┐
│   sentence-transformers     │  ← embeds your question into a vector
│   (all-MiniLM-L6-v2)        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│         ChromaDB            │  ← cosine similarity search
│   (122 DSA knowledge chunks)│    returns top 4 relevant chunks
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│     Prompt Assembly         │  ← retrieved chunks + conversation
│                             │    history (last 10 turns) + depth
│  [Retrieved Context]        │    instruction injected into prompt
│  [Chat History]             │
│  [User Question]            │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│          Ollama             │  ← local LLM generates the answer
│       (llama3.2 etc.)       │    streamed token by token
└─────────────┬───────────────┘
              │
              ▼
         Streamlit UI
```

**On first launch**, all `.txt` files in `knowledge_base/` are chunked by section, embedded, and stored in `chroma_db/` on disk. This takes ~15 seconds and only happens once.

---

## Knowledge Base Topics

- Arrays & Strings
- Binary Search
- Linked Lists
- Stacks & Queues
- Trees
- Graphs
- Sorting Algorithms
- Dynamic Programming
- Hashing
- Recursion & Backtracking

---

## Setup

### 1. Install Ollama
Download from **[ollama.com/download](https://ollama.com/download)** and run the app.

### 2. Pull a model
```bash
ollama serve          # keep this running in a terminal
ollama pull llama3.2  # ~2 GB download
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Project Structure

```
├── app.py                        # Streamlit UI and app entry point
├── requirements.txt
├── rag/
│   ├── loader.py                 # Loads and chunks knowledge base .txt files
│   ├── vector_store.py           # ChromaDB wrapper with sentence-transformer embeddings
│   └── retriever.py              # Enriches query with chat history, runs similarity search
├── chatbot/
│   └── client.py                 # Builds prompt, streams response from Ollama
└── knowledge_base/
    └── *.txt                     # DSA notes split into sections (122 chunks total)
```

---

## Model Options

Select any of these from the sidebar (run `ollama pull <name>` first):

| Model | Size | Notes |
|---|---|---|
| `llama3.2` | 2 GB | Fast, good quality — recommended |
| `mistral` | 4.1 GB | Strong at code explanations |
| `phi3` | 2.2 GB | Fastest, lightest |
| `llama3.1` | 4.7 GB | Best reasoning |
| `gemma2` | 5.4 GB | Good instruction following |
