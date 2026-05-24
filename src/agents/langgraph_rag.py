from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.corrective_agent import fallback_answer_node, grade_retrieved_chunks
from src.agents.retriever_agent import VectorStore, build_vector_store, retriever_node
from src.agents.summary_agent import summary_node
from src.agents.supervisor_agent import supervisor_node
from src.data_preprocessing.preparation import ensure_data_assets
from src.models import ProjectSettings
from src.settings import load_project_settings


class GraphState(TypedDict, total=False):
    question: str
    chat_history: list[dict[str, str]]
    supervisor_decision: str
    supervisor_message: str
    retrieved_context: str
    retrieved_chunks: list[dict[str, Any]]
    retrieved_chunks_relevant: bool
    retrieval_grade_message: str
    answer: str
    source_path: str


def build_graph(settings: ProjectSettings, store: VectorStore):
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", lambda state: supervisor_node(state, settings))
    graph.add_node("retrieve", lambda state: retriever_node(state, settings, store))
    graph.add_node("grade_docs", lambda state: grade_retrieved_chunks(state, settings))
    graph.add_node("summarize", lambda state: summary_node(state, settings))
    graph.add_node("fallback", lambda state: fallback_answer_node(state, settings))
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", lambda state: state.get("supervisor_decision", "retrieve"), {"retrieve": "retrieve", "clarify": END, "reject": END})
    graph.add_edge("retrieve", "grade_docs")
    graph.add_conditional_edges("grade_docs", lambda state: "summarize" if state.get("retrieved_chunks_relevant", False) else "fallback", {"summarize": "summarize", "fallback": "fallback"})
    graph.add_edge("summarize", END)
    graph.add_edge("fallback", END)
    return graph.compile(checkpointer=MemorySaver())


def create_app_resources(project_root: Path | None = None) -> tuple[ProjectSettings, VectorStore, Any, Path]:
    root = project_root or Path(__file__).resolve().parents[2]
    settings = load_project_settings(root)
    ensure_data_assets(settings)
    store = build_vector_store(settings)
    graph = build_graph(settings, store)
    return settings, store, graph, store.bundle.source_path


def default_thread_id() -> str:
    return "streamlit-session"

