import io
import argparse
from pathlib import Path

from PIL import Image

from src.agents.langgraph_rag import create_app_resources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the LangGraph agentic graph as a PNG.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_graph/model_graph.png"),
        help="PNG output path.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, _, graph, _ = create_app_resources(args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    graph_image_data = graph.get_graph().draw_mermaid_png()
    image = Image.open(io.BytesIO(graph_image_data))
    image.save(args.output)
    print(f"Saved graph PNG to {args.output}")


if __name__ == "__main__":
    main()