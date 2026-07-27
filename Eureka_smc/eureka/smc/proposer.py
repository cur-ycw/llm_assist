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
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["TaskContext", "extract_reward_code", "EurekaReflectionProposer", "FakeProposer"]


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

    def __init__(self, ctx: TaskContext, openai_module: Any):
        self.ctx = ctx
        self._openai = openai_module  # 注入 openai 模块，便于测试替换
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_calls = 0

    # ---- LLM 调用（带重试，镜像官方 eureka.py:89-115）----
    def _chat(self, messages: list[dict], n: int) -> list[str]:
        chunk = n
        contents: list[str] = []
        while len(contents) < n:
            resp = None
            for attempt in range(1000):
                try:
                    resp = self._openai.ChatCompletion.create(
                        model=self.ctx.model,
                        messages=messages,
                        temperature=self.ctx.temperature,
                        n=min(chunk, n - len(contents)),
                    )
                    break
                except Exception as e:  # noqa: BLE001 — 与官方一致，粗粒度重试
                    if attempt >= 10:
                        chunk = max(chunk // 2, 1)
                    logger.info(f"LLM attempt {attempt + 1} failed: {e}")
                    time.sleep(1)
            if resp is None:
                raise RuntimeError("LLM call failed after too many attempts")
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
        messages = [
            {"role": "system", "content": self.ctx.initial_system},
            {"role": "user", "content": self.ctx.initial_user},
            {"role": "assistant", "content": f"```python\n{parent_code}\n```"},
            {"role": "user", "content": parent_feedback + self.ctx.code_output_tip},
        ]
        return extract_reward_code(self._chat(messages, 1)[0])


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
