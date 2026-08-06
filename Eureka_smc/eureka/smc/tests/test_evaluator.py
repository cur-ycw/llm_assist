"""Isaac Gym evaluator 的超时、清理与共享 task 锁测试。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from eureka.smc.evaluator import IsaacGymEvalConfig, IsaacGymEvaluator
from eureka.smc.score import ScoreConfig


PROMPTS = {
    "policy_feedback": "feedback {epoch_freq}",
    "code_feedback": " code",
    "code_output_tip": " tip",
    "execution_error_feedback": "error {traceback_msg}",
}


def _config(tmp_path: Path, **overrides) -> IsaacGymEvalConfig:
    values = {
        "isaac_root_dir": str(tmp_path / "isaac"),
        "eureka_root_dir": str(tmp_path / "eureka"),
        "task": "Ant",
        "suffix": "GPT",
        "env_name": "ant",
        "task_code_string": "def compute_reward(self):\n    pass\n",
        "output_file": str(tmp_path / "tasks" / "antgpt.py"),
        "max_iterations": 500,
        "startup_timeout_seconds": 1.0,
        "training_timeout_seconds": 1.0,
        "lock_timeout_seconds": 1.0,
    }
    values.update(overrides)
    return IsaacGymEvalConfig(**values)


def _evaluator(tmp_path: Path, **overrides) -> IsaacGymEvaluator:
    return IsaacGymEvaluator(
        ScoreConfig(metric="gt_reward", fallback_metric="gt_reward", aggregate="final_window_mean"),
        seeds=[42], prompts=PROMPTS, eval_cfg=_config(tmp_path, **overrides), cache=False,
    )


class _TimeoutProcess:
    def __init__(self):
        self.terminated = False
        self.killed = False
        self.wait_calls = []

    def communicate(self, timeout=None):
        raise subprocess.TimeoutExpired("train.py", timeout)

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return 0

    def kill(self):
        self.killed = True


def test_config_salt_changes_when_timeout_policy_changes(tmp_path):
    base = _evaluator(tmp_path, startup_timeout_seconds=10.0)
    changed = _evaluator(tmp_path, startup_timeout_seconds=20.0)

    assert base._config_salt() != changed._config_salt()


def test_collect_training_timeout_terminates_process_and_records_reason(tmp_path):
    evaluator = _evaluator(tmp_path, training_timeout_seconds=3.0)
    log = tmp_path / "train_seed42.txt"
    log.write_text("training output\n")
    proc = _TimeoutProcess()

    logs, stdout_path, error = evaluator._collect_seed(proc, str(log), {})

    assert logs is None
    assert stdout_path == str(log)
    assert error == "training_timeout after 3.0s"
    assert proc.terminated is True
    assert "[SMC evaluator timeout] training_timeout after 3.0s" in log.read_text()


def test_wait_for_training_start_times_out(tmp_path, monkeypatch):
    evaluator = _evaluator(tmp_path, startup_timeout_seconds=0.01)
    log = tmp_path / "train_seed42.txt"
    log.write_text("still booting\n")
    ticks = iter((0.0, 0.02))
    monkeypatch.setattr("eureka.smc.evaluator.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("eureka.smc.evaluator.time.sleep", lambda _: None)

    with pytest.raises(TimeoutError, match="training_start_timeout"):
        evaluator._wait_for_training_start(str(log), {})


def test_shared_task_lock_times_out_when_held(tmp_path):
    primary = _evaluator(tmp_path, lock_timeout_seconds=1.0)
    contender = _evaluator(tmp_path, lock_timeout_seconds=0.01)

    with primary._task_write_lock():
        with pytest.raises(TimeoutError, match="shared task lock"):
            with contender._task_write_lock():
                pass


def test_invalid_timeout_configuration_rejected(tmp_path):
    with pytest.raises(ValueError, match="timeouts must be positive"):
        _config(tmp_path, startup_timeout_seconds=0)
