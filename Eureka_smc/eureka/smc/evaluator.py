"""奖励代码评估器（迁移计划 §4.2、§5.2、§6.9）。

统一接口 ``Evaluator.evaluate(reward_code, candidate_id, artifact_dir) -> EvalRecord``：
在冻结的 shared seed panel 上评估一段奖励代码，返回唯一的归一化 ``search_score``、结构化
反馈与复现工件。带 code-hash 缓存（相同代码 + 完整 panel + 配置复用同一聚合结果）。

  * ``FakeEvaluator``：便宜的测试替身，从 ``# quality=`` 标记读分并加 seed 噪声，供集成
    测试在不烧 GPU 的前提下验证 SMC 逻辑；
  * ``IsaacGymEvaluator``：真实评估器，抽取官方 ``eureka.py`` 的注入→训练→tensorboard
    解析→反馈构造流程。对 ``utils`` 采用**延迟导入**（真实运行时 cwd 在 ``eureka/``），
    因此本模块可在无 Isaac Gym 的环境中被导入，只要不实例化该类。

无效候选（不可解析/导入/执行）一律 ``search_score=None``、``valid=False``，绝不用有限失败
分（计划 §4.2）；是否重采样或拒绝由 island 层决定。
"""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .particle import EvalRecord
from .score import ScoreConfig, compute_search_score

logger = logging.getLogger(__name__)

__all__ = ["Evaluator", "FakeEvaluator", "IsaacGymEvaluator", "IsaacGymEvalConfig"]


def _cache_key(reward_code: str, salt: str, seeds: Sequence[int]) -> str:
    h = hashlib.sha256()
    h.update(reward_code.encode())
    h.update(b"\x00")
    h.update(salt.encode())
    h.update(b"\x00")
    h.update(",".join(map(str, seeds)).encode())
    return h.hexdigest()


def _chunks(seq: list, size: int):
    """把 seq 切成每块至多 size 个的连续波次（size<=0 视为一整波）。"""
    if size is None or size <= 0:
        yield list(seq)
        return
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


class Evaluator:
    """评估器基类，实现 code-hash 缓存；子类实现 ``_evaluate_uncached``。"""

    def __init__(self, score_cfg: ScoreConfig, seeds: Sequence[int], cache: bool = True):
        self.score_cfg = score_cfg
        self.seeds = tuple(seeds)
        self._cache: dict[str, EvalRecord] = {}
        self._cache_enabled = cache
        self.n_evals = 0  # 真实执行（未命中缓存）次数

    def _config_salt(self) -> str:
        return ""  # 子类覆盖：纳入所有影响评估结果的配置

    def evaluate(self, reward_code: str, candidate_id: str,
                 artifact_dir: Optional[Path] = None) -> EvalRecord:
        key = _cache_key(reward_code, self._config_salt(), self.seeds)
        if self._cache_enabled and key in self._cache:
            rec = self._cache[key]
            logger.info(f"[eval] cache hit for {candidate_id} ({key[:8]})")
            return rec
        rec = self._evaluate_uncached(reward_code, candidate_id, artifact_dir)
        rec.cache_key = key
        rec.train_seeds = self.seeds
        self.n_evals += 1
        if self._cache_enabled:
            self._cache[key] = rec
        return rec

    def _evaluate_uncached(self, reward_code, candidate_id, artifact_dir) -> EvalRecord:
        raise NotImplementedError

    def evaluate_batch(self, items: Sequence[tuple]) -> list[EvalRecord]:
        """批量评估。默认**串行**兜底：逐个走带缓存的 ``evaluate``。

        供 ``FakeEvaluator`` 等测试替身使用——保持与旧逐粒子路径完全一致的 RNG 消费顺序
        （每个 evaluator 的 rng 仍按 0..N-1 顺序被调用）、缓存命中与 ``n_evals`` 语义，因此
        island 改批量后 Fake 集成测试的确定性不变。``IsaacGymEvaluator`` 覆盖为分波并发。
        ``items`` 为 ``(reward_code, candidate_id, artifact_dir)`` 列表，返回对齐的记录。
        """
        return [self.evaluate(code, cid, d) for code, cid, d in items]


# --------------------------------------------------------------------------- fake

class FakeEvaluator(Evaluator):
    """确定性测试替身：``# quality=q`` → search_score = clip(q + seed 噪声)。

    对 panel 内每个 seed 加独立高斯噪声后取均值，模拟随机 RL evaluator；不含
    ``# quality`` 标记（或空代码）视为无效候选。RNG 由外部注入以保证可复现。
    """

    def __init__(self, score_cfg, seeds, rng, noise: float = 0.03, cache: bool = True):
        super().__init__(score_cfg, seeds, cache)
        self.rng = rng
        self.noise = noise

    def _evaluate_uncached(self, reward_code, candidate_id, artifact_dir) -> EvalRecord:
        m = re.search(r"# quality=([0-9.]+)", reward_code or "")
        if m is None or "def " not in (reward_code or ""):
            return EvalRecord(search_score=None, valid=False, executable=False,
                              feedback="Invalid candidate: no parseable reward.",
                              error="fake:invalid")
        base = float(m.group(1))
        per_seed = np.clip(base + self.noise * self.rng.normal(size=len(self.seeds)), 0.0, None)
        score = float(per_seed.mean())
        return EvalRecord(
            search_score=score, valid=True, executable=True,
            feedback=f"Fake feedback: mean search_score={score:.4f} over {len(self.seeds)} seeds.",
            raw_metrics={"fake_quality": base, "search_score": score},
            metadata={"per_seed": per_seed.tolist()},
        )


# ----------------------------------------------------------------------- isaacgym

class IsaacGymEvalConfig:
    """IsaacGymEvaluator 的运行参数（对应官方 eureka.py 的 subprocess 参数）。"""

    def __init__(self, *, isaac_root_dir: str, eureka_root_dir: str, task: str, suffix: str,
                 env_name: str, task_code_string: str, output_file: str,
                 max_iterations: int, use_wandb: bool = False, wandb_username: str = "",
                 wandb_project: str = "", capture_video: bool = False):
        self.isaac_root_dir = isaac_root_dir
        self.eureka_root_dir = eureka_root_dir
        self.task = task
        self.suffix = suffix
        self.env_name = env_name
        self.task_code_string = task_code_string  # 已做 task->task+suffix 替换
        self.output_file = output_file
        self.max_iterations = max_iterations
        self.use_wandb = use_wandb
        self.wandb_username = wandb_username
        self.wandb_project = wandb_project
        self.capture_video = capture_video


class IsaacGymEvaluator(Evaluator):
    """真实评估器：注入奖励代码 → 训练 → 解析 tensorboard → 构造反馈。

    ``evaluate``（单个）仍走共享 ``output_file`` 的路径；``evaluate_batch`` 做**伪并行**：
    一个驱动进程按“写 output_file → Popen train.py → block_until_training（等它 import 完，
    非训练结束）→ 写下一个”把一波候选铺开，靠 ``block_until_training`` 门控共享文件写入，
    再统一 ``communicate`` 收集。``set_freest_gpu`` 每次挑最空卡 → 自动摊到多张 GPU。每波
    至多 ``max_concurrent`` 个子进程并发，给显存留余量。
    """

    def __init__(self, score_cfg, seeds, prompts: Mapping[str, str],
                 eval_cfg: IsaacGymEvalConfig, cache: bool = True, max_concurrent: int = 6):
        super().__init__(score_cfg, seeds, cache)
        self.prompts = prompts
        self.cfg = eval_cfg
        self.max_concurrent = max_concurrent  # 每波并发子进程上限（3 卡下 6≈每卡 2 个）

    def _config_salt(self) -> str:
        c = self.cfg
        return f"{c.task}|{c.suffix}|{c.max_iterations}|{c.env_name}"

    # ---- 延迟导入 utils（真实运行时 cwd 在 eureka/）----
    @staticmethod
    def _utils():
        from utils.extract_task_code import get_function_signature
        from utils.file_utils import load_tensorboard_logs
        from utils.misc import block_until_training, filter_traceback, set_freest_gpu
        return dict(get_function_signature=get_function_signature,
                    load_tensorboard_logs=load_tensorboard_logs,
                    block_until_training=block_until_training,
                    filter_traceback=filter_traceback, set_freest_gpu=set_freest_gpu)

    def _inject(self, code_string: str, u) -> Optional[str]:
        """把奖励代码注入 task 环境源码，返回完整环境代码；签名解析失败返回 None。"""
        try:
            gpt_sig, _ = u["get_function_signature"](code_string)
        except Exception:
            return None
        sig_lines = [
            f"self.rew_buf[:], self.rew_dict = {gpt_sig}",
            "self.extras['gpt_reward'] = self.rew_buf.mean()",
            "for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
        ]
        indent = " " * 8
        block = "\n".join(indent + ln for ln in sig_lines)
        tcs = self.cfg.task_code_string
        if "def compute_reward(self)" in tcs:
            env = tcs.replace("def compute_reward(self):", "def compute_reward(self):\n" + block)
        elif "def compute_reward(self, actions)" in tcs:
            env = tcs.replace("def compute_reward(self, actions):",
                              "def compute_reward(self, actions):\n" + block)
        else:
            raise NotImplementedError("task code lacks a known compute_reward signature")
        body = code_string if "@torch.jit.script" in code_string else "@torch.jit.script\n" + code_string
        return (env + "\nfrom typing import Tuple, Dict\nimport math\nimport torch\n"
                "from torch import Tensor\n" + body + "\n")

    def _launch_seed(self, env_code: str, seed: int, artifact_dir: Path, u):
        """写共享 output_file → 启动 train.py → block_until_training（等它 import 完）。

        返回 (proc, rl_filepath)。调用者必须在写下一个候选的 output_file **之前**完成本次
        block_until_training（已在此函数内），从而共享文件写入被串起来、训练本身并发。
        """
        u["set_freest_gpu"]()
        with open(self.cfg.output_file, "w") as f:
            f.write(env_code)
        rl_filepath = str(artifact_dir / f"train_seed{seed}.txt")
        c = self.cfg
        with open(rl_filepath, "w") as f:
            proc = subprocess.Popen(
                ["python", "-u", f"{c.isaac_root_dir}/train.py", "hydra/output=subprocess",
                 f"task={c.task}{c.suffix}", f"wandb_activate={c.use_wandb}",
                 f"wandb_entity={c.wandb_username}", f"wandb_project={c.wandb_project}",
                 f"headless={not c.capture_video}", f"capture_video={c.capture_video}",
                 "force_render=False", f"max_iterations={c.max_iterations}", f"seed={seed}"],
                stdout=f, stderr=f)
        u["block_until_training"](rl_filepath)
        return proc, rl_filepath

    @staticmethod
    def _collect_seed(proc, rl_filepath: str, u) -> tuple[Optional[dict], str, str]:
        """等一个已启动的 train.py 跑完并解析，返回 (tensorboard_logs|None, stdout_path, tb)。"""
        proc.communicate()
        with open(rl_filepath) as f:
            stdout_str = f.read()
        tb = u["filter_traceback"](stdout_str)
        if tb != "":
            return None, rl_filepath, tb
        tb_dir = ""
        for line in stdout_str.split("\n"):
            if line.startswith("Tensorboard Directory:"):
                tb_dir = line.split(":")[-1].strip()
                break
        if not tb_dir:
            return None, rl_filepath, "No Tensorboard Directory in stdout."
        return u["load_tensorboard_logs"](tb_dir), rl_filepath, ""

    def _build_feedback(self, logs: Mapping) -> str:
        """镜像官方 eureka.py:243-279 的反馈构造。"""
        max_iters = np.array(logs["gt_reward"]).shape[0]
        epoch_freq = max(int(max_iters // 10), 1)
        content = self.prompts["policy_feedback"].format(epoch_freq=epoch_freq)
        for metric in logs:
            if "/" in metric:
                continue
            series = ["{:.2f}".format(x) for x in logs[metric][::epoch_freq]]
            mx, mn = max(logs[metric]), min(logs[metric])
            mean = sum(logs[metric]) / len(logs[metric])
            if metric in ("gt_reward", "gpt_reward"):
                if "consecutive_successes" not in logs:
                    content += (f"ground-truth score: {series}, Max: {mx:.2f}, "
                                f"Mean: {mean:.2f}, Min: {mn:.2f} \n")
            else:
                name = "task_score" if metric == "consecutive_successes" else metric
                content += f"{name}: {series}, Max: {mx:.2f}, Mean: {mean:.2f}, Min: {mn:.2f} \n"
        return content + self.prompts["code_feedback"] + self.prompts["code_output_tip"]

    def _signature_error_record(self, t0: float) -> EvalRecord:
        return EvalRecord(
            search_score=None, valid=False, executable=False,
            feedback=self.prompts["execution_error_feedback"].format(
                traceback_msg="Cannot parse reward function signature! Re-write a new one.")
            + self.prompts["code_output_tip"],
            error="signature_parse_error", wall_time_s=time.time() - t0)

    def _aggregate(self, seed_results: dict, artifact_dir: Path, t0: float) -> EvalRecord:
        """把一个候选各 seed 的运行结果聚合成 EvalRecord（原 _evaluate_uncached 尾部逻辑）。"""
        env_copy = str(artifact_dir / "env_code.py")
        per_seed_norm: list[float] = []
        gpt_rewards: list[float] = []
        raw_first: Optional[float] = None
        feedback = ""
        stdout_paths: list[str] = []
        last_tb = ""
        for seed in self.seeds:
            logs, stdout_path, tb = seed_results.get(seed, (None, "", "no_result"))
            stdout_paths.append(stdout_path)
            if logs is None:
                last_tb = tb
                continue
            sc = compute_search_score(logs, self.score_cfg)
            if sc is None:
                continue
            raw, norm = sc
            per_seed_norm.append(norm)
            if "gpt_reward" in logs and len(logs["gpt_reward"]):
                gpt_rewards.append(float(np.mean(logs["gpt_reward"])))
            if not feedback:  # 用首个成功 seed 构造反馈
                feedback = self._build_feedback(logs)
                raw_first = raw

        if not per_seed_norm:
            return EvalRecord(
                search_score=None, valid=False, executable=False,
                feedback=self.prompts["execution_error_feedback"].format(traceback_msg=last_tb)
                + self.prompts["code_output_tip"],
                error=last_tb or "no_valid_metric", env_code_path=env_copy,
                stdout_paths=tuple(stdout_paths), wall_time_s=time.time() - t0)

        score = float(np.mean(per_seed_norm))  # panel 聚合：归一化分均值（计划 §5.2）
        return EvalRecord(
            search_score=score, valid=True, executable=True, feedback=feedback,
            raw_metrics={"raw_first_seed": raw_first, "search_score": score,
                         "gpt_reward_mean": float(np.mean(gpt_rewards)) if gpt_rewards else None},
            reward_components={"per_seed_normalized": per_seed_norm},
            env_code_path=env_copy, stdout_paths=tuple(stdout_paths),
            wall_time_s=time.time() - t0)

    def _run_candidates(self, specs: list[tuple]) -> dict:
        """伪并行执行一批候选（specs=[(key, code, artifact_dir)]），返回 {key: EvalRecord}。

        对每个候选注入奖励代码；把所有 (候选×seed) 展平成 launch unit，按 max_concurrent
        分波：一波内串行 launch（写共享 output_file→Popen→block_until_training 门控），再统一
        collect。写入被 block_until_training 串起来，训练本身并发；set_freest_gpu 摊到多卡。
        不处理缓存/n_evals（由调用方负责）。
        """
        u = self._utils()
        env_by_key: dict = {}
        dir_by_key: dict = {}
        t0_by_key: dict = {}
        recs: dict = {}
        for key, code, d in specs:
            d = Path(d) if d else Path(f"candidate_{key[:8]}")
            d.mkdir(parents=True, exist_ok=True)
            dir_by_key[key], t0_by_key[key] = d, time.time()
            env_code = self._inject(code, u)
            env_by_key[key] = env_code
            if env_code is None:
                recs[key] = self._signature_error_record(t0_by_key[key])
            else:
                with open(d / "env_code.py", "w") as f:
                    f.write(env_code)

        runnable = [key for key, _, _ in specs if env_by_key[key] is not None]
        seed_results: dict = {key: {} for key in runnable}
        units = [(key, seed) for key in runnable for seed in self.seeds]
        for wave in _chunks(units, self.max_concurrent):
            launched = []
            for key, seed in wave:  # 串行 launch：写→启动→等 import（门控共享文件）
                proc, rl = self._launch_seed(env_by_key[key], seed, dir_by_key[key], u)
                launched.append((key, seed, proc, rl))
            for key, seed, proc, rl in launched:  # 统一 collect：并发训练在此汇合
                seed_results[key][seed] = self._collect_seed(proc, rl, u)

        for key in runnable:
            recs[key] = self._aggregate(seed_results[key], dir_by_key[key], t0_by_key[key])
        return recs

    def _evaluate_uncached(self, reward_code, candidate_id, artifact_dir) -> EvalRecord:
        """单候选评估（heldout / 无 batch 路径用）：多 seed 也会分波并发。"""
        artifact_dir = Path(artifact_dir) if artifact_dir else Path(f"candidate_{candidate_id}")
        key = _cache_key(reward_code, self._config_salt(), self.seeds)
        return self._run_candidates([(key, reward_code, artifact_dir)])[key]

    def evaluate_batch(self, items: Sequence[tuple]) -> list[EvalRecord]:
        """伪并行批量评估。命中缓存的直接复用；未命中的按 code 去重后一起分波并发。

        与基类串行兜底同接口（items=[(code, cid, artifact_dir)]），但 IsaacGym 下把整批
        （及各自的 seed panel）铺开到多卡并发。缓存写入、``n_evals`` 累加在此完成，语义与
        单个 ``evaluate`` 一致（每个**唯一未命中** code 记 1 次真实评估）。
        """
        results: list[Optional[EvalRecord]] = [None] * len(items)
        specs_by_key: dict = {}          # key -> (code, artifact_dir)  去重
        idx_by_key: dict = {}            # key -> [item indices]
        for i, (code, cid, d) in enumerate(items):
            key = _cache_key(code, self._config_salt(), self.seeds)
            if self._cache_enabled and key in self._cache:
                results[i] = self._cache[key]
                continue
            idx_by_key.setdefault(key, []).append(i)
            specs_by_key.setdefault(key, (code, Path(d) if d else Path(f"candidate_{cid}")))

        if specs_by_key:
            specs = [(key, code, d) for key, (code, d) in specs_by_key.items()]
            computed = self._run_candidates(specs)
            for key, rec in computed.items():
                rec.cache_key = key
                rec.train_seeds = self.seeds
                self.n_evals += 1
                if self._cache_enabled:
                    self._cache[key] = rec
                for i in idx_by_key[key]:
                    results[i] = rec
        return results  # type: ignore[return-value]
