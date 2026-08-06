from __future__ import annotations

import json
from pathlib import Path

from eureka.smc.run_manifest import write_run_manifest


def test_write_run_manifest_persists_protocol_identity(tmp_path):
    path = tmp_path / "run_manifest.json"
    manifest = write_run_manifest(
        path,
        resolved_config={"algo": {"budget": 80}, "env": {"task": "Cartpole"}},
        task="Cartpole",
        env_name="cartpole",
        search_seeds=[42],
        validation_seeds=[200, 201],
        test_seeds=[300, 301],
        budget={"n_particles": 16, "mutation_budget": 64, "budget_total": 80},
        checkpoint={"enabled": True, "path": "checkpoint.json"},
        project_root=Path(__file__).resolve().parents[3],
    )

    payload = json.loads(path.read_text())
    assert payload == manifest
    assert payload["task"] == "Cartpole"
    assert payload["seed_panels"] == {
        "search": [42], "validation": [200, 201], "test": [300, 301],
    }
    assert payload["budget"]["budget_total"] == 80
    assert payload["resolved_config"]["algo"]["budget"] == 80
    assert "git_commit" in payload
    assert "git_status_porcelain" in payload
