import ollama
from typing import Generator

DEPTH_INSTRUCTIONS = {
    "Beginner": (
        "You are teaching someone who is new to DSA. Use simple language, real-world analogies, "
        "and step-by-step explanations. Avoid heavy jargon — define any technical term you use. "
        "Include intuitive examples. Keep explanations clear and encouraging."
    ),
    "Intermediate": (
        "You are teaching someone with basic programming knowledge who understands fundamentals. "
        "Use proper CS terminology. Provide code examples in Python. Discuss time and space complexity. "
        "Explain the 'why' behind design choices."
    ),
    "Advanced": (
        "You are teaching an experienced developer preparing for senior-level interviews or competitive programming. "
        "Discuss trade-offs, edge cases, and optimizations. Reference algorithmic theory where relevant. "
        "Provide rigorous complexity analysis. Mention alternative approaches and when to prefer each. "
        "Code examples should be clean and production-quality."
    ),
}

SYSTEM_BASE = """You are Study Buddy, an expert DSA (Data Structures and Algorithms) tutor. \
Your goal is to give accurate, helpful, and well-structured answers to DSA questions.

Guidelines:
- Use the provided knowledge base context to ground your answers.
- If the context does not cover the question, draw from your own knowledge and say so.
- For follow-up questions, refer to the conversation history to maintain continuity.
- Always be concise but complete — don't pad answers unnecessarily.
- Format code blocks with ```python (or the relevant language).
- When asked to compare two approaches, use a clear structure (pros/cons or table).
"""


class StudyBuddyClient:
    MAX_HISTORY_TURNS = 10

    def chat_stream(
        self,
        user_message: str,
        retrieved_context: str,
        conversation_history: list[dict],
        depth: str = "Intermediate",
        model: str = "llama3.2",
    ) -> Generator[str, None, None]:
        """
        Stream the response from Ollama token by token.
        Yields text chunks as they arrive — use with st.write_stream().
        """
        system_prompt = SYSTEM_BASE + "\n\n" + DEPTH_INSTRUCTIONS.get(depth, DEPTH_INSTRUCTIONS["Intermediate"])

        if retrieved_context:
            augmented_message = f"{retrieved_context}\n\n---\n\n**User Question:** {user_message}"
        else:
            augmented_message = user_message

        trimmed = self._trim_history(conversation_history)

        messages = (
            [{"role": "system", "content": system_prompt}]
            + trimmed
            + [{"role": "user", "content": augmented_message}]
        )

        stream = ollama.chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            yield chunk["message"]["content"]

    def _trim_history(self, history: list[dict]) -> list[dict]:
        max_messages = self.MAX_HISTORY_TURNS * 2
        return history[-max_messages:] if len(history) > max_messages else history
