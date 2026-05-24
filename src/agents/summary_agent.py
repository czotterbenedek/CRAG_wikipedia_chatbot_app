from typing import Any

import ollama

from src.conversation_memory import build_relevant_history_block
from src.models import ProjectSettings


def _chat(model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()


def summarize_answer(state: dict[str, Any], settings: ProjectSettings) -> dict[str, Any]:
    question = state.get("question", "")
    history = state.get("chat_history", [])
    retrieved_context = state.get("retrieved_context", "")
    history_block = build_relevant_history_block(
        history,
        question,
        settings.embedding_model,
        max_turns=settings.history_max_turns,
        similarity_threshold=settings.history_similarity_threshold,
    )
    user_prompt = f"""
<conversation_history>
{history_block if history_block else '<turn role="system">No prior history.</turn>'}
</conversation_history>

<user_question>
{question}
</user_question>

<retrieved_context>
{retrieved_context if retrieved_context else '<chunk id="none">No chunks were retrieved.</chunk>'}
</retrieved_context>

Answer using only the retrieved context.
""".strip()
    answer = _chat(settings.summary.model, settings.summary.system_prompt, user_prompt, settings.summary.temperature)
    return {**state, "answer": answer}


def summary_node(state: dict[str, Any], settings: ProjectSettings) -> dict[str, Any]:
    return summarize_answer(state, settings)
