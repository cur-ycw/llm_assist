from __future__ import annotations

import json
from pathlib import Path

import pytest

from eureka.smc.checkpoint import CheckpointError, load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_and_version(tmp_path: Path):
    path = tmp_path / "run" / "state.json"
    state = {"budget": 7, "rng_state": {"state": [1, 2, 3]}, "archive": ["abc"]}
    assert save_checkpoint(path, state) == path
    assert load_checkpoint(path) == state
    assert json.loads(path.read_text())["version"] == 1


def test_checkpoint_replace_is_complete(tmp_path: Path):
    path = tmp_path / "state.json"
    save_checkpoint(path, {"generation": 1})
    save_checkpoint(path, {"generation": 2, "particles": []})
    assert load_checkpoint(path) == {"generation": 2, "particles": []}
    assert not list(tmp_path.glob(".state.json.*.tmp"))


def test_checkpoint_rejects_wrong_version(tmp_path: Path):
    path = tmp_path / "state.json"
    save_checkpoint(path, {"x": 1}, version=2)
    with pytest.raises(CheckpointError, match="version"):
        load_checkpoint(path)
    assert load_checkpoint(path, expected_version=2) == {"x": 1}


def test_checkpoint_rejects_malformed_json(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text("not json")
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_checkpoint_does_not_use_pickle(tmp_path: Path):
    with pytest.raises(TypeError):
        save_checkpoint(tmp_path / "state.json", {"not_json": {1, 2}})
