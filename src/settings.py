from pathlib import Path
from typing import Any

import yaml

from src.models import CorrectiveSettings, FallbackSettings, GlobalSettings, ProjectSettings, SummarySettings, SupervisorSettings


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_project_settings(project_root: Path) -> ProjectSettings:
    conf_dir = project_root / "conf"
    return ProjectSettings(
        project_root=project_root,
        global_settings=GlobalSettings.model_validate(_read_yaml(conf_dir / "config.yml")),
        supervisor=SupervisorSettings.model_validate(_read_yaml(conf_dir / "supervisor_agent_config.yml")),
        summary=SummarySettings.model_validate(_read_yaml(conf_dir / "summary_agent_config.yml")),
        corrective=CorrectiveSettings.model_validate(_read_yaml(conf_dir / "corrective_agent_config.yml")),
        fallback=FallbackSettings.model_validate(_read_yaml(conf_dir / "fallback_agent_config.yml")),
    )
