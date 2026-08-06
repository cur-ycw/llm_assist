"""Write a self-contained manifest for one SMC-Eureka run."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["write_run_manifest"]


def _git_value(args: Sequence[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=cwd, text=True, capture_output=True,
            check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def write_run_manifest(
    path: str | Path,
    *,
    resolved_config: Mapping[str, Any],
    task: str,
    env_name: str,
    search_seeds: Sequence[int],
    validation_seeds: Sequence[int],
    test_seeds: Sequence[int],
    budget: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    """Persist the immutable run identity before search resources are consumed."""
    project_root = Path(project_root)
    path = Path(path)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(path.parent.resolve()),
        "project_root": str(project_root.resolve()),
        "git_commit": _git_value(["rev-parse", "HEAD"], project_root),
        "git_status_porcelain": _git_value(["status", "--porcelain"], project_root),
        "task": task,
        "env_name": env_name,
        "seed_panels": {
            "search": [int(seed) for seed in search_seeds],
            "validation": [int(seed) for seed in validation_seeds],
            "test": [int(seed) for seed in test_seeds],
        },
        "budget": dict(budget),
        "checkpoint": dict(checkpoint),
        "resolved_config": dict(resolved_config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest
