"""SMC 运行的 JSONL 事件日志（迁移计划 §6、§8 Phase 5）。

每个 proposal / eval / resample / stage / summary 写一行 JSON，用于事后还原任意粒子的
父代、提议、评估与接受决策。追加写、逐行 flush，便于中断后检查。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

__all__ = ["EventLogger"]


class EventLogger:
    def __init__(self, path: str | Path, clock: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)  # 行缓冲
        self._seq = 0
        self._clock = clock

    def log(self, event: str, **fields: Any) -> None:
        rec: dict[str, Any] = {"seq": self._seq, "event": event}
        if self._clock:
            rec["t"] = time.time()
        rec.update(fields)
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._seq += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *exc: Optional[object]) -> None:
        self.close()
