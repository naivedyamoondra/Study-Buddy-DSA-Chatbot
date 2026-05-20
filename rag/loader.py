from pathlib import Path


def load_documents(knowledge_base_dir: str) -> list[dict]:
    documents = []
    for txt_file in sorted(Path(knowledge_base_dir).glob("*.txt")):
        topic = txt_file.stem.replace("_", " ").title()
        documents.extend(_split_into_chunks(txt_file.read_text(encoding="utf-8"), topic))
    return documents


def _split_into_chunks(content: str, topic: str) -> list[dict]:
    chunks = []
    current_section = "Overview"
    current_lines: list[str] = []
    chunk_index = 0

    def flush():
        nonlocal chunk_index
        text = "\n".join(current_lines).strip()
        if len(text) > 50:
            chunks.append({
                "id": f"{topic}_{chunk_index}",
                "text": text,
                "topic": topic,
                "section": current_section,
            })
            chunk_index += 1

    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## "):
            flush()
            current_section = stripped[3:]
            current_lines = [f"[Topic: {topic} | Section: {current_section}]"]
        elif stripped == "---":
            flush()
            current_lines = [f"[Topic: {topic} | Section: {current_section}]"]
        else:
            current_lines.append(line)

    flush()
    return chunks
