import importlib
from pathlib import Path
from typing import Any

st: Any = importlib.import_module("streamlit")

from src.agents.langgraph_rag import create_app_resources, default_thread_id


@st.cache_resource(show_spinner=True)
def load_app():
    return create_app_resources(Path(__file__).resolve().parent)


def run_chat_turn(graph, question: str, chat_history: list[dict[str, str]], thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = {
        "question": question,
        "chat_history": chat_history,
    }
    return graph.invoke(state, config=config)


def main() -> None:
    st.set_page_config(page_title="Wikipedia CRAG Q&A", page_icon="🤖", layout="wide")
    _, _, graph, _ = load_app()

    st.title("🤖 Wikipedia CRAG Q&A ChatBot")
    st.markdown("Ask questions about general knowledge!")
    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"][:3], 1):
                        st.markdown(
                            f"**{i}. Chunk {source['chunk_id']}** from `{source['source']}`  \n"
                            f"Score: `{source['score']:.3f}`"
                        )
                        st.write(source["text"])
                        st.divider()

    if len(st.session_state.messages) > 0:
        st.caption(f"{len(st.session_state.messages) // 2} exchanges in this session")

    if prompt := st.chat_input("Ask a question ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching with full conversation context..."):
                try:
                    result = run_chat_turn(
                        graph,
                        prompt,
                        st.session_state.messages,
                        default_thread_id(),
                    )

                    if result.get("supervisor_decision") in {"clarify", "reject"}:
                        assistant_text = result.get("supervisor_message", "I need more information.")
                    else:
                        assistant_text = result.get("answer", "I could not generate an answer.")

                    sources = result.get("retrieved_chunks", []) if result.get("retrieved_chunks_relevant", True) else []
                except Exception as error:
                    assistant_text = f"Error: {error}"
                    sources = []

            st.write(assistant_text)

            if sources:
                with st.expander("View Sources"):
                    for i, source in enumerate(sources[:3], 1):
                        st.markdown(
                            f"**{i}. Chunk {source['chunk_id']}** from `{source['source']}`  \n"
                            f"Score: `{source['score']:.3f}`"
                        )
                        st.write(source["text"])
                        st.divider()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_text,
                "sources": sources,
            }
        )

        if len(st.session_state.messages) // 2 > 1:
            st.caption("Chain is using full conversation history automatically")


if __name__ == "__main__":
    main()
