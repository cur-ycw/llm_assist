"""版本化、原子化的 SMC 运行状态 checkpoint。

checkpoint 仅使用 JSON，避免加载 pickle 时执行任意代码的风险。调用方应把可恢复的
普通状态（预算、粒子快照、RNG state、archive 索引等）放入一个 JSON 对象。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

__all__ = [
    "CHECKPOINT_VERSION",
    "CheckpointError",
    "save_checkpoint",
    "load_checkpoint",
]


CHECKPOINT_VERSION = 1


class CheckpointError(ValueError):
    """checkpoint 不能安全读取或版本不兼容时抛出。"""


def _fsync_directory(directory: Path) -> None:
    """在支持的 POSIX 文件系统上同步 rename 元数据。"""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def save_checkpoint(
    path: str | Path,
    state: Mapping[str, Any],
    *,
    version: int = CHECKPOINT_VERSION,
) -> Path:
    """原子保存版本化 JSON checkpoint，并返回目标路径。

    内容先写入目标目录内的临时文件并 fsync，再用 ``os.replace`` 发布，因而中断
    时读者只会看到上一次完整 checkpoint 或这一次完整 checkpoint。
    """
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint state must be a mapping")
    if not isinstance(version, int) or version < 1:
        raise ValueError("checkpoint version must be a positive integer")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": version, "state": dict(state)}
    tmp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as handle:
            tmp_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
        _fsync_directory(target.parent)
    except (OSError, TypeError, ValueError) as exc:
        if tmp_name is not None:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
        if isinstance(exc, (TypeError, ValueError)):
            raise TypeError("checkpoint state must be JSON serializable") from exc
        raise
    return target


def load_checkpoint(
    path: str | Path,
    *,
    expected_version: Optional[int] = CHECKPOINT_VERSION,
) -> dict[str, Any]:
    """安全加载 checkpoint 的状态对象，并验证版本和基本形状。"""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"cannot load checkpoint {path!s}") from exc

    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint payload must be an object")
    version = payload.get("version")
    state = payload.get("state")
    if not isinstance(version, int) or not isinstance(state, dict):
        raise CheckpointError("checkpoint must contain integer version and object state")
    if expected_version is not None and version != expected_version:
        raise CheckpointError(
            f"checkpoint version {version} does not match expected version {expected_version}"
        )
    return state
