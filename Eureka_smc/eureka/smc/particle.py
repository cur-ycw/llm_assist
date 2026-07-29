"""SMC 粒子与评估记录的数据模型（迁移计划 §4，首版精简）。

首版只覆盖单岛 + 单一 ``eureka_reflection`` proposal 所需字段；RF-Agent 五操作相关
字段（``donor_ids``、``proposal_action``、``aligned_thought`` 等）留到后续批次再加。

关键不变量（计划 §4.2）：
  * 无效候选（不可解析/导入/执行）的 ``search_score`` 必须为 ``None``，绝不能用
    有限的失败分（如 -10000）——否则会污染 ESS 温度二分。
  * ``proposal_parent_id`` 指向真正的 LLM 提议来源；
  * ``accepted_transition_parent_id`` 只记录“真正改变并被 MH 接受”的代码状态，
    用于 accepted-edit genealogy（路径推理时回溯它，跳过 clone/reject）；
  * 重采样 clone 只写 ``clone_ancestor_id``，不得伪造一次代码优化。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class EvalRecord:
    """一次奖励代码评估的统一封装（计划 §4.2）。

    ``search_score`` 是用于 SMC 权重与 MH 接受率的**唯一**标量；对无效候选为
    ``None``。其余字段供反馈构造、日志和复现使用。
    """

    search_score: Optional[float]           # 归一化后的搜索分；无效 -> None
    valid: bool                             # 可解析 + 可导入 + 可执行
    executable: bool                        # 训练子进程无 traceback
    feedback: str = ""                      # 喂给下一次 proposal 的反馈文本
    raw_metrics: dict[str, float] = field(default_factory=dict)  # 原始标量（未归一化）
    reward_components: dict[str, Any] = field(default_factory=dict)  # 分量时间序列/摘要
    error: Optional[str] = None             # traceback（若有）
    train_seeds: tuple[int, ...] = ()       # 本次评估使用的 shared seed panel
    tensorboard_dirs: tuple[str, ...] = ()
    stdout_paths: tuple[str, ...] = ()
    env_code_path: Optional[str] = None     # 注入后的完整环境代码路径
    wall_time_s: float = 0.0
    cache_key: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["train_seeds"] = list(self.train_seeds)
        d["tensorboard_dirs"] = list(self.tensorboard_dirs)
        d["stdout_paths"] = list(self.stdout_paths)
        return d


@dataclass
class RewardParticle:
    """一个奖励函数候选（SMC 粒子）。"""

    id: str
    reward_code: str
    eval: EvalRecord
    island_id: int = 0
    generation: int = 0
    # provenance（DAG 的首版子集）
    proposal_parent_id: Optional[str] = None            # LLM 提议来源
    accepted_transition_parent_id: Optional[str] = None  # 真正被接受的代码状态父
    clone_ancestor_id: Optional[str] = None             # 重采样 clone 来源
    proposal_action: Optional[str] = None               # RF-Agent action（generic 路径为 None）
    design_thought: Optional[str] = None                # 提议时 LLM 先写的一句设计意图（免额外调用）
    artifact_dir: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- 便捷只读属性（分数唯一真源在 eval 上）----
    @property
    def search_score(self) -> Optional[float]:
        return self.eval.search_score

    @property
    def valid(self) -> bool:
        return self.eval.valid

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "island_id": self.island_id,
            "generation": self.generation,
            "proposal_parent_id": self.proposal_parent_id,
            "accepted_transition_parent_id": self.accepted_transition_parent_id,
            "clone_ancestor_id": self.clone_ancestor_id,
            "proposal_action": self.proposal_action,
            "design_thought": self.design_thought,
            "artifact_dir": str(self.artifact_dir) if self.artifact_dir else None,
            "search_score": self.search_score,
            "valid": self.valid,
            "reward_code": self.reward_code,
            "eval": self.eval.to_json(),
            "metadata": self.metadata,
        }
