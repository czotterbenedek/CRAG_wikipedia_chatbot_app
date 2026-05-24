from pathlib import Path

from src.data_preprocessing.preparation import prepare_data_assets
from src.settings import load_project_settings


def main() -> None:
    project_root = Path(__file__).resolve().parent
    settings = load_project_settings(project_root)
    source_path = prepare_data_assets(settings)
    print(f"Prepared data assets for {source_path.name}")


if __name__ == "__main__":
    main()