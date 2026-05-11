# MARL Modular Reward Agent 技术路线报告

## 1. 当前框架的最终定位

当前框架已经不适合再表述为“优化奖励函数本身”，更准确的定义应为：

**在 cooperative MARL 中，构造一个由 LLM 初始化的 modular reward agent，通过共享 reward modules 和 per-agent contextual selector，为不同 agent 在不同局部情境下动态提供辅助激励。**

换句话说：
- 这不是直接学习新的环境主奖励函数；
- 这也不是简单给每个 agent 一个静态额外奖励；
- 而是学习一个 **agent-specific, context-sensitive auxiliary incentive mechanism**。

---

## 2. 为什么从 MTRL 转向 MARL

最初尝试从 MTRL 出发，但逐步发现以下问题：

1. 不同 task 虽然 observation 维度可能一致，但语义并不一致；
2. 想让 LLM 在多任务 raw state 上直接生成共享 reward code，语义对齐困难极大；
3. 为每个 task 构造统一 primitive space 或 shared reward code 的工程量过高，且第一版难以验证；
4. 这种设定下，LLM 需要理解过多环境细节、状态构造、任务差异，上下文负担过重。

相比之下，MARL 更适合当前思路：

1. 多 agent 处于同一个环境语义空间中，不再有跨 task 的状态语义漂移；
2. 不同 agent 的差异主要来自 **局部观测视角不同**，而不是任务定义不同；
3. 因此 selector 更自然地可以被定义为：
   - 当前 agent 在当前局部情境下，应该激活哪些奖励原则；
4. LLM 不需要解析多任务 raw obs 语义，而只需处理：
   - 单环境规则
   - 协作目标
   - agent 局部交互模式
   - reward module 的反馈报告

所以当前最合理的主线是：

**LLM4RL + cooperative MARL + modular auxiliary reward agent**

---

## 3. 核心问题定义

### 3.1 要解决的不是“奖励函数学习”

当前方法不直接替代环境奖励，而是在原始 team reward 基础上增加 per-agent 的辅助激励：

\[
r_{i,t}^{train} = r_t^{env} + \beta r_{i,t}^{aux}
\]

其中：
- \(r_t^{env}\)：环境提供的全局团队奖励；
- \(r_{i,t}^{aux}\)：面向 agent \(i\) 的辅助奖励；
- \(\beta\)：辅助奖励强度。

因此当前方法本质上是在解决：

### 3.2 目标问题

**在 cooperative MARL 中，面对稀疏、延迟且全局统一的团队奖励，如何为每个 agent 在不同局部情境下动态分配更有效的辅助激励，以提升探索、信用分配和协作形成效率？**

---

## 4. 动机：这篇文章真正解决什么问题

当前框架最稳的动机应建立在以下三个问题上。

### 4.1 全局 team reward 过于粗糙

在 cooperative MARL 中，环境通常只提供一个全局团队奖励：
- 稀疏；
- 延迟；
- 混合了多个 agent 的综合效果；
- 不能反映不同 agent 当前局部处境的差异。

因此，不同 agent 很难从统一 team reward 中知道：
- 当前该探索还是配合；
- 当前该推进还是规避冲突；
- 当前该执行局部支持还是完成最终目标。

### 4.2 现有 credit assignment 主要在“分责任”，不在“给激励”

像 COMA、difference rewards、coalition credit 等方法主要回答的是：
- 某个 agent 对全局回报贡献了多少；

但它们不一定能回答：
- 当前这个 agent 在当前局部情境下最该被哪类行为原则驱动。

当前方法的核心切口是：

**从 credit assignment 转向 incentive design。**

### 4.3 现有 per-agent auxiliary reward 方法通常是 monolithic scalar

已有一些方法会为每个 agent 学一个单独奖励，但通常直接输出：

\[
r_{i,t}^{aux} = f_\theta(\cdot)
\]

问题在于：
- 黑盒；
- 缺乏结构；
- 难解释；
- 难复用；
- 无法清楚表达“当前为什么激活某种激励”。

因此，本框架希望回答：

**能否把 per-agent auxiliary reward 建模为一组共享奖励原则的动态组合，而不是一个单一标量？**

---

## 5. 当前方法的形式化定义

### 5.1 Dec-POMDP 设定

考虑 cooperative Dec-POMDP：

\[
\mathcal G = \langle \mathcal S, \{\mathcal O_i\}_{i=1}^n, \{\mathcal A_i\}_{i=1}^n, P, r^{env}, \gamma \rangle
\]

其中：
- \(n\)：agent 数量；
- \(s_t\)：全局状态；
- \(o_i^t\)：agent \(i\) 的局部观测；
- \(a_i^t\)：agent \(i\) 的动作；
- \(r_t^{env}\)：环境全局团队奖励。

每个 agent 维护局部历史：

\[
h_i^t = (o_i^1, a_i^1, \dots, o_i^t)
\]

并执行 decentralized policy：

\[
a_i^t \sim \pi_i(a \mid o_i^t, h_i^t)
\]

训练仍然采用 CTDE。

---

### 5.2 Shared reward module pool

定义共享模块池：

\[
\mathcal M = \{m_1, m_2, \dots, m_K\}
\]

每个模块 \(m_k\) 表示一种可复用的行为激励原则，例如：
- 鼓励接近关键区域；
- 避免与队友冲突；
- 保持覆盖；
- 支援队友；
- 稳定推进局部目标。

模块输入是 agent 当前上下文特征：

\[
x_i^t = \phi(o_i^t, h_i^t, c_i^t)
\]

其中：
- \(\phi\)：agent-context encoder；
- \(c_i^t\)：可选的邻域关系、局部通信或训练时可见上下文。

模块输出：

\[
m_k(x_i^t) \in \mathbb R
\]

---

### 5.3 Contextual selector

selector 输出每个模块在当前局部情境下的权重：

\[
w_i^t = \sigma_\psi(x_i^t)
\]

其中：
- \(w_i^t \in \mathbb R^K\)；
- 可采用 softmax 或 top-k sparse gating。

于是，agent-specific auxiliary reward 定义为：

\[
r_{i,t}^{aux} = \sum_{k=1}^{K} w_{i,k}^t \, m_k(x_i^t)
\]

最终训练奖励：

\[
r_{i,t}^{train} = r_t^{env} + \beta r_{i,t}^{aux}
\]

评估时仍然只看：

\[
J_{eval} = \mathbb E\left[\sum_t \gamma^t r_t^{env}\right]
\]

这意味着：
- auxiliary reward 只用于训练；
- 方法目标不是改写环境最终目标，而是提供更适合的训练激励。

---

## 6. LLM 在当前框架中的角色

当前框架里，LLM 不适合：
- 在线读取 raw obs/action；
- 每步参与 reward 预测；
- 直接通过梯度进入主训练回路。

当前最合理的角色有两个。

### 6.1 初始化 reward modules

LLM 根据：
- 环境规则；
- 协作目标；
- agent 交互模式；
- reward module schema / DSL；

生成一组初始 reward modules。

### 6.2 外循环 refinement

训练一段时间后，收集模块表现摘要，交给 LLM 做：
- keep
- patch
- rescale
- split
- merge
- append
- disable

因此，当前方法中的 LLM 更像：

**reward-module generator + refiner**

而不是 reward predictor。

---

## 7. selector 怎么训练

这是当前方法的核心难点之一。

### 7.1 关键原则

selector 不能直接用自己生成的 shaped reward 来训练自己，否则会出现自指问题：
- selector 学到的是“什么让额外奖励变大”；
- 而不是“什么真的提高了原始环境表现”。

因此采用双层优化视角：

### 7.2 Inner loop

固定当前 reward modules 和 selector，使用：

\[
r_{i,t}^{train} = r_t^{env} + \beta r_{i,t}^{aux}
\]

训练 MARL policy。

### 7.3 Outer loop

每隔一个训练窗口（如若干 episodes / 一定 env steps），用原始环境奖励评估 policy 改进：

\[
\Delta_u = J_u^{env} - J_{u-1}^{env}
\]

或和 no-shaping / shadow baseline 比较，得到 selector 的反馈信号。

### 7.4 推荐第一版做法

第一版最稳的 selector 训练方式：
- soft selector；
- window-level outer update；
- 不先做离散硬采样；
- 加入稀疏性、平滑性、熵正则。

推荐总原则：

**policy 看 shaped reward 学；selector 看原始环境性能提升学。**

---

## 8. reward modules 怎么评估好坏

这是当前框架的第二个核心难点。

reward modules 的质量不能只看最终 return，而应拆成四层评估。

### 8.1 Executability

检查：
- 是否能正常运行；
- 是否输出 NaN / Inf；
- 是否数值范围稳定；
- 是否不是常数。

建议指标：
- valid rate
- nan rate
- mean / std / min / max
- active rate

### 8.2 Usage

检查 selector 是否真正使用该模块：
- average weight
- peak weight
- activation frequency
- agent-wise usage histogram
- context-wise usage pattern

### 8.3 Utility

核心指标：

\[
U_k = J(\mathcal M) - J(\mathcal M \setminus \{m_k\})
\]

即 leave-one-out module utility。

也可以结合：
- when-high-weight-selected, future team return delta；
- coordination metrics 改善；
- sample efficiency gain。

### 8.4 Redundancy / Interference

检查：
- 模块之间输出相关性；
- 共激活模式；
- 是否互相冲突或高度重复。

所以总体原则是：

**模块质量 = 可执行性 + 使用情况 + utility + 冗余性分析**

---

## 9. reward modules 怎么更新

### 9.1 关键判断

如果 reward module 是 LLM 生成的 code / DSL，它本身通常不直接吃梯度。

因此：
- selector 通过 outer-loop learning 更新；
- module 通过 evaluator + LLM feedback 更新；
- 不应把 module 当成普通神经网络 head 直接做 end-to-end SGD。

### 9.2 Outer-loop refinement

每隔若干训练窗口，对每个模块生成诊断摘要：
- executability
- usage
- utility
- redundancy
- failure patterns

LLM 根据这些摘要输出：
- keep
- patch
- rescale
- split
- merge
- append
- disable

得到新的模块池：

\[
\mathcal M^{(u+1)} = \text{LLMRefine}(\mathcal M^{(u)}, F^{(u)})
\]

### 9.3 稳定更新策略

不要直接 hard replace，推荐：
- shadow evaluation
- append-then-prune
- residual patch

避免 selector 与 policy 因 module pool 剧烈变化而崩溃。

---

## 10. 完整训练 pipeline（推荐 MVP 版本）

### Phase 0: module initialization
1. LLM 生成 K 个初始 reward modules；
2. validator 过滤不可执行或数值异常模块。

### Phase 1: policy warm-up
3. 仅用环境原始 team reward 训练一段时间；
4. 让 policy 有基本行为，降低后续 selector 学习噪声。

### Phase 2: inner-loop training
5. 固定当前模块池；
6. 每步为每个 agent 构造上下文特征 \(x_i^t\)；
7. selector 输出模块权重 \(w_i^t\)；
8. 组合 auxiliary reward \(r_{i,t}^{aux}\)；
9. 用 \(r_t^{env} + \beta r_{i,t}^{aux}\) 更新 MARL policy。

### Phase 3: outer-loop selector update
10. 每个窗口结束后，仅用原始环境奖励评估当前 policy；
11. 用环境表现提升作为 selector 的学习信号。

### Phase 4: module evaluation
12. 对每个模块计算 executability、usage、utility、redundancy；
13. 生成模块诊断摘要。

### Phase 5: LLM refinement
14. 定期把模块摘要交给 LLM；
15. LLM 对模块池执行 keep/patch/split/merge/append/disable；
16. validator 检查通过后进入下一轮训练。

一句话概括：

**policy 用 shaped reward 学，selector 用原始环境改进学，reward modules 用 evaluator + LLM 外循环学。**

---

## 11. 与已有工作的核心区别

当前方法既不同于传统 cooperative MARL baseline，也不同于已有的 per-agent reward 方法。

### 11.1 和标准 cooperative MARL 的区别
如 QMIX / QPLEX / MAPPO：
- 只使用原始 team reward；
- 不为 agent 提供结构化个体化激励。

### 11.2 和 policy specialization 方法的区别
如 IQL / CDS / ROMA / RODE：
- 主要在 policy / Q-network 层做个体化与分化；
- 当前方法在 **reward / incentive layer** 做个体化。

### 11.3 和 reward-side 方法的区别
如 COMA / difference rewards / IRAT / CenRA：
- 许多现有方法直接为 agent 分配单一标量辅助奖励；
- 当前方法强调：
  - shared reward module pool；
  - per-agent contextual selector；
  - agent × time/context 维度上的 incentive routing；
  - reward principles 的动态组合，而不是 monolithic scalar。

一句话差异：

**现有方法多在学“给每个 agent 多少奖励”；当前方法更关注“当前该用哪类奖励原则来驱动该 agent”。**

---

## 12. 当前最推荐的 baseline 家族

如果后续要做实验，baseline 应至少覆盖三类。

### 12.1 标准 cooperative MARL
- QMIX / QPLEX
- 或 MAPPO

### 12.2 policy-side specialization
- IQL
- CDS
- ROMA / RODE（可二选一）

### 12.3 reward-side baselines
- COMA
- Difference rewards / Dr.Reinforce
- IRAT
- PRD
- CenRA
- monolithic per-agent reward head

最小核心套餐建议：
- QPLEX 或 QMIX
- IQL
- CDS
- monolithic per-agent reward head
- CenRA
- Ours

---

## 13. 当前最值得做的 ablation

### 13.1 No selector
固定模块权重，不做动态选择。

### 13.2 No modules
直接输出 monolithic per-agent scalar reward。

### 13.3 No sharing
每个 agent 自己维护一套模块池。

### 13.4 Static selector
每个 episode / trajectory 固定一次 selector，不做 per-step switching。

### 13.5 Random selector
测试 selector 是否真的学到了有意义的情境路由。

---

## 14. 当前框架的风险与难点

### 14.1 selector credit 太慢
selector 不直接控制动作，只影响训练激励，因此学习信号天然滞后。

### 14.2 module-policy co-adaptation
module pool 与 policy 同时更新容易导致系统不稳定。

### 14.3 reward hacking / misalignment
错误模块可能让 agent 学会追求 auxiliary reward 而偏离全局目标。

### 14.4 module utility 估计成本高
leave-one-out、在线消融、冗余分析都很昂贵。

### 14.5 LLM 修补可能引入不可控变化
所以必须依赖：
- structured schema / DSL
- validator
- shadow evaluation

---

## 15. 推荐的后续推进顺序

### Stage 1: 先做非 LLM 的最小闭环
- 固定手工 reward modules
- soft selector
- outer-loop selector training
- policy backbone 跑通

目标：证明“shared modules + contextual selector”本身是否成立。

### Stage 2: 引入 LLM 初始化模块
- 让 LLM 生成初始模块池
- 仍先不做 refinement

目标：证明 LLM prior 是否比手工初始模块更好。

### Stage 3: 引入 evaluator
- 完整统计 executability / usage / utility / redundancy

目标：模块反馈闭环完整化。

### Stage 4: 引入 LLM outer-loop refinement
- patch / split / merge / append / disable
- 慢时间尺度更新模块池

目标：把“LLM-initialized”升级为“LLM-refined”。

---

## 16. 这篇工作的最稳表述

### 一句话主线
**We study an LLM-guided modular reward agent for cooperative MARL, where a shared pool of reward modules is dynamically routed to individual agents according to their local interaction context, providing agent-specific auxiliary incentives under sparse and delayed team rewards.**

### 中文版
**我们研究 cooperative MARL 中的 LLM 引导式模块化 reward agent：通过共享奖励模块池，并根据每个 agent 的局部交互情境动态路由奖励模块，在稀疏且延迟的团队奖励下为其提供个体化辅助激励。**

---

## 17. 最后的判断

当前框架最符合事实的定位不是：
- reward function optimization
- cross-task reward code generation

而是：
- modular reward agent
- adaptive incentive design
- LLM-initialized and LLM-refined auxiliary reward mechanism for cooperative MARL

如果后续在其他工作目录继续拓展，建议始终保持这条主线，不再混用“优化奖励函数”表述。