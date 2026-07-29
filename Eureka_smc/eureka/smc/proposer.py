"""奖励代码提议器（迁移计划 §6.4：单一 eureka_reflection proposal）。

首版只提供一个隐式混合修改策略的通用反思 proposal：把父粒子的当前代码 + 训练反馈 +
分数喂给 LLM，由其自行诊断并输出完整新代码。与原 Eureka 的区别在于——父代不是全局
唯一 best，而是 SMC 重采样后**每个粒子各自的当前状态**（计划 §6.4）。

本模块含三部分：
  * ``extract_reward_code``：从 LLM 响应里抽取奖励函数代码（复用官方 Eureka 的 regex
    + 去 import + 截到首个 ``def`` 的逻辑），纯字符串处理、可单测；
  * ``EurekaReflectionProposer``：真实 LLM 提议器（OpenAI Chat Completions，带重试与
    token 计数）；
  * ``FakeProposer``：确定性测试替身，不调用 LLM，供 FakeEvaluator 集成测试用。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["TaskContext", "extract_reward_code", "extract_design_thought",
           "EurekaReflectionProposer", "FakeProposer"]


_CODE_PATTERNS = [
    r"```python(.*?)```",
    r"```(.*?)```",
    r'"""(.*?)"""',
    r'""(.*?)""',
    r'"(.*?)"',
]


def extract_reward_code(response_content: str) -> Optional[str]:
    """从 LLM 响应中提取奖励函数代码字符串（官方 Eureka eureka.py:129-149 的等价实现）。

    依次尝试多种代码围栏；成功后去掉围栏前的多余 import，截取到首个 ``def``。若最终不含
    函数定义则返回 ``None``（视为无效提议）。
    """
    code_string: Optional[str] = None
    for pattern in _CODE_PATTERNS:
        m = re.search(pattern, response_content, re.DOTALL)
        if m is not None:
            code_string = m.group(1).strip()
            break
    if code_string is None:
        code_string = response_content

    # 截到首个 def（去掉前置多余 import/说明）
    lines = code_string.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            code_string = "\n".join(lines[i:])
            break
    else:
        return None  # 无函数定义
    return code_string


def extract_design_thought(response_content: str) -> Optional[str]:
    """取代码围栏之前的自然语言文本，作为该次提议的一句 design thought。

    action 提示要求 LLM「先用一句话说明将做什么改动，再写完整代码」，因此 thought 就在
    首个 ``` 围栏之前。免额外 LLM 调用；无围栏或围栏在开头则返回 None。
    """
    idx = response_content.find("```")
    if idx <= 0:
        return None
    thought = response_content[:idx].strip()
    return thought or None


@dataclass
class TaskContext:
    """一次 SMC 运行内冻结的任务级提示上下文。"""

    initial_system: str          # 已 format（含 reward signature + code_output_tip）
    initial_user: str            # 已 format（含 obs 代码 + 任务描述）
    code_output_tip: str
    model: str
    temperature: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


class EurekaReflectionProposer:
    """真实 LLM 提议器：初始批量生成 + 单粒子反思重生成。"""

    def __init__(self, ctx: TaskContext, openai_module: Any, max_attempts: int = 30,
                 action_prompts: Optional[dict[str, str]] = None):
        self.ctx = ctx
        self._openai = openai_module  # 注入 openai 模块，便于测试替换
        self.max_attempts = max_attempts  # 单次调用的最大重试次数（避免无限重试卡死）
        # RF-Agent action 名 → 指令尾文本；generic 模式为空 dict（不影响 reflect 路径）。
        self.action_prompts = action_prompts or {}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_calls = 0
        self._counter_lock = threading.Lock()  # reflect_batch 并发下保护计数器

    # ---- LLM 调用（带重试，镜像官方 eureka.py:89-115）----
    def _chat(self, messages: list[dict], n: int) -> list[str]:
        chunk = n
        contents: list[str] = []
        while len(contents) < n:
            resp = None
            for attempt in range(self.max_attempts):
                try:
                    resp = self._openai.ChatCompletion.create(
                        model=self.ctx.model,
                        messages=messages,
                        temperature=self.ctx.temperature,
                        n=min(chunk, n - len(contents)),
                    )
                    break
                except Exception as e:  # noqa: BLE001 — 粗粒度重试
                    msg = str(e)
                    # 账户余额/额度不足属不可恢复错误，快速失败并给清晰提示，
                    # 避免像官方那样对 403 无限重试刷屏卡死。
                    if any(k in msg for k in ("余额", "balance", "insufficient", "quota")):
                        raise RuntimeError(f"LLM 额度/余额不足，请充值或更换 key：{msg[:160]}")
                    if attempt >= 10:
                        chunk = max(chunk // 2, 1)
                    logger.info(f"LLM attempt {attempt + 1} failed: {e}")
                    time.sleep(1)
            if resp is None:
                raise RuntimeError(f"LLM call failed after {self.max_attempts} attempts")
            with self._counter_lock:  # reflect_batch 线程池并发时保护累加
                self.n_calls += 1
                self.total_prompt_tokens += resp["usage"]["prompt_tokens"]
                self.total_completion_tokens += resp["usage"]["completion_tokens"]
            contents.extend(c["message"]["content"] for c in resp["choices"])
        return contents[:n]

    def initial_batch(self, n: int) -> list[Optional[str]]:
        """用冻结初始提示生成 ``n`` 个初始奖励代码（计划 §6.1）。"""
        messages = [
            {"role": "system", "content": self.ctx.initial_system},
            {"role": "user", "content": self.ctx.initial_user},
        ]
        return [extract_reward_code(c) for c in self._chat(messages, n)]

    def reflect(self, parent_code: str, parent_feedback: str) -> Optional[str]:
        """基于父粒子当前代码 + 训练反馈生成 1 个新代码（计划 §6.4）。

        复刻 Eureka 的四消息范式，但父代是该粒子自身状态而非全局 best。
        """
        return extract_reward_code(self._chat(self._reflect_messages(
            parent_code, parent_feedback), 1)[0])

    def _reflect_messages(self, parent_code: str, parent_feedback: str) -> list[dict]:
        """Eureka 四消息反思范式（父代=该粒子自身状态）。reflect / reflect_batch 共用。"""
        return [
            {"role": "system", "content": self.ctx.initial_system},
            {"role": "user", "content": self.ctx.initial_user},
            {"role": "assistant", "content": f"```python\n{parent_code}\n```"},
            {"role": "user", "content": parent_feedback + self.ctx.code_output_tip},
        ]

    def reflect_batch(self, items: list[tuple[str, str]]) -> list[Optional[str]]:
        """并发发出多个**各不相同**的 reflect 提议（伪并行：线程池 + 每个独立 HTTP）。

        ``items`` 为 ``(parent_code, parent_feedback)`` 列表；返回与之对齐的新代码列表
        （某项 LLM 失败/无法解析 → 该位置为 ``None``，由 island 视作 no-op 保留父代）。
        初始 16 个用单次 ``n=16`` 调用已并行；这里解决每阶段 16 个**不同 prompt** 的反思
        本来只能串行发的问题。计数器已加锁，线程安全。
        """
        if not items:
            return []

        def work(it: tuple[str, str]) -> Optional[str]:
            try:
                return extract_reward_code(self._chat(
                    self._reflect_messages(it[0], it[1]), 1)[0])
            except Exception as e:  # noqa: BLE001 — 单个反思失败降级为 no-op，不拖垮整阶段
                logger.warning(f"reflect_batch item failed (视作 no-op): {str(e)[:160]}")
                return None

        results: list[Optional[str]] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=len(items)) as ex:
            futs = {ex.submit(work, it): i for i, it in enumerate(items)}
            for fut in futs:
                results[futs[fut]] = fut.result()
        return results

    # ---- RF-Agent action 提议（Phase-3a：mutation_structure / mutation_parameter）----
    def _action_messages(self, parent_code: str, parent_feedback: str, action: str) -> list[dict]:
        """四消息范式，但第 4 条 user 尾追加 action 专用指令。

        指令置于 parent_feedback 之后、code_output_tip 之前——作为模型看到输出格式提醒前的
        最后一条约束，主导本次改动方向（结构 or 仅参数）。action 未在 action_prompts 中登记
        时指令为空串，退化为普通反思。
        """
        instr = self.action_prompts.get(action, "")
        tail = parent_feedback + ("\n" + instr if instr else "") + self.ctx.code_output_tip
        return [
            {"role": "system", "content": self.ctx.initial_system},
            {"role": "user", "content": self.ctx.initial_user},
            {"role": "assistant", "content": f"```python\n{parent_code}\n```"},
            {"role": "user", "content": tail},
        ]

    def propose_batch(
        self, items: list[tuple[str, str, str]]
    ) -> list[tuple[Optional[str], Optional[str]]]:
        """并发发出 action 路由后的提议（伪并行：线程池 + 每个独立 HTTP）。

        ``items`` 为 ``(parent_code, parent_feedback, action)`` 列表；返回与之对齐的
        ``(reward_code, design_thought)`` 列表。某项失败/无法解析 → ``(None, None)``（island
        视作 no-op 保留父代）。design_thought 从同一响应的代码围栏前文本抽取，不额外调 LLM。
        """
        if not items:
            return []

        def work(it: tuple[str, str, str]) -> tuple[Optional[str], Optional[str]]:
            parent_code, parent_feedback, action = it
            try:
                content = self._chat(
                    self._action_messages(parent_code, parent_feedback, action), 1)[0]
                return extract_reward_code(content), extract_design_thought(content)
            except Exception as e:  # noqa: BLE001 — 单个提议失败降级为 no-op，不拖垮整阶段
                logger.warning(f"propose_batch item failed (视作 no-op): {str(e)[:160]}")
                return None, None

        results: list[tuple[Optional[str], Optional[str]]] = [(None, None)] * len(items)
        with ThreadPoolExecutor(max_workers=len(items)) as ex:
            futs = {ex.submit(work, it): i for i, it in enumerate(items)}
            for fut in futs:
                results[futs[fut]] = fut.result()
        return results


class FakeProposer:
    """确定性测试替身：不调用 LLM，生成携带“质量”标记的合成代码。

    约定：代码里嵌入 ``# quality=<float>`` 注释，``FakeEvaluator`` 据此打分。初始批量在
    ``[base, base+spread]`` 均匀取值；``reflect`` 在父质量上加一个正偏移（模拟改进），并按
    ``noop_rate`` 概率原样返回父代码（模拟 no-op 提议）。RNG 由外部注入以保证可复现。
    """

    _TEMPLATE = "def compute_reward(self):  # quality={q:.6f}\n    return self.rew_buf, {{}}\n"

    def __init__(self, rng, base: float = 0.2, spread: float = 0.5,
                 improve: float = 0.08, noop_rate: float = 0.1):
        self.rng = rng
        self.base, self.spread = base, spread
        self.improve, self.noop_rate = improve, noop_rate
        self.n_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @staticmethod
    def _quality_of(code: str) -> float:
        m = re.search(r"# quality=([0-9.]+)", code)
        return float(m.group(1)) if m else 0.0

    def initial_batch(self, n: int) -> list[Optional[str]]:
        self.n_calls += n
        qs = self.base + self.spread * self.rng.random(n)
        return [self._TEMPLATE.format(q=q) for q in qs]

    def reflect(self, parent_code: str, parent_feedback: str) -> Optional[str]:
        self.n_calls += 1
        if self.rng.random() < self.noop_rate:
            return parent_code  # no-op 提议
        q = self._quality_of(parent_code) + self.improve * self.rng.normal(1.0, 0.6)
        return self._TEMPLATE.format(q=max(0.0, q))

    def propose_batch(
        self, items: list[tuple[str, str, str]]
    ) -> list[tuple[Optional[str], Optional[str]]]:
        """确定性 action 提议兜底：忽略 action 语义（Fake 无领域概念），复用 reflect 的质量
        改进逻辑，并回一个带 action 标签的合成 design_thought，供 rf 路径集成测试断言。"""
        out: list[tuple[Optional[str], Optional[str]]] = []
        for parent_code, parent_feedback, action in items:
            new_code = self.reflect(parent_code, parent_feedback)
            if new_code is None or new_code == parent_code:
                out.append((new_code, None))  # None / no-op：无 thought
            else:
                out.append((new_code, f"fake design thought for {action}"))
        return out
