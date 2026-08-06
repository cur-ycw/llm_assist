# SMC-Eureka 当前实现状态

更新时间：2026-08-06

本文记录当前 `/root/ycw/Eureka_smc` 分支上，按照“预算与进展自适应 SMC 奖励搜索”计划已经完成的工程部分、尚未完成的部分、当前验证结果和下一步顺序。

> 本文只描述 SMC-Eureka 相关代码。工作区中 `pymarl/`、`smac_test/` 以及其他实验目录的大量删除/未跟踪文件属于既有工作区噪音，本轮不处理、不恢复、不清理。

## 一、已完成的核心实现

### 1. 单岛 SMC 主体框架

已具备单岛 `SMCIsland` 搜索流程，核心路径为：

```text
初始化有效粒子
  -> 剩余预算计算
  -> KL/rESS 父代选择分布
  -> multinomial 重采样
  -> LLM proposal
  -> candidate evaluator
  -> sigmoid reward-only 接受
  -> 父代槽位回写
  -> 下一轮
```

对应文件：

- `eureka/smc/island.py`
- `eureka/smc/resampling.py`
- `eureka/smc/proposer.py`
- `eureka/smc/evaluator.py`

当前仍保持单岛设计，migration 默认关闭；没有提前引入多岛、Thompson selector 或消融矩阵。

### 2. 新预算模型已落地

当前正式配置使用 Isaac Gym 目标资源模型：

```text
初始生成预算       = 16
每轮 mutation 子代 = 8
mutation 轮数       = 8
后续 mutation 预算  = 8 × 8 = 64
总搜索预算          = 16 + 64 = 80
```

配置文件：`eureka/cfg/algo/smc.yaml`

```yaml
n_particles: 16
children_per_round: 8
init_budget: 16
mutation_rounds: 8
mutation_budget: 64
budget: 80
```

`SMCIslandConfig.__post_init__()` 已增加以下校验：

- `init_budget == n_particles`；
- 显式配置时 `mutation_budget == mutation_rounds * children_per_round`；
- `budget == init_budget + mutation_budget`；
- `budget` 不得小于初始生成预算；
- `k_min` 必须位于 `[1, n_particles]`；
- `eta` 必须为有限非负数。

未显式指定完整拆分时，旧 Fake 测试仍可根据剩余预算推导轮数，并允许最后一轮使用 partial mutation budget；正式 YAML 使用完整的 16 + 64 严格拆分。

### 3. N 槽位固定语义已实现

当前种群语义已经从“每轮 population 缩为 child 数量”改为固定父代槽位：

- population 始终保持 `N=16` 个槽位；
- 每轮从完整 N 个槽位计算 `q`；
- 每轮抽取 `M=8` 个修改机会，允许同一父代被重复抽中；
- 被接受的 child 回写到对应来源父代的原始槽位；
- 未获得资源或被拒绝的槽位保留原粒子；
- 下一轮继续从 N 个槽位计算控制分布。

Fake 测试已经将旧的 population shrink 断言改为 N-slot invariant 断言。

### 4. Full rESS 控制器已实现

`eureka/smc/kl_controller.py` 已从旧的 entropy/KL 热路径改为二阶矩相对 ESS 控制：

```text
rESS(q) = 1 / (N * sum(q**2))
```

已经加入或更新：

- `effective_parents(q) = 1 / sum(q**2)`；
- `relative_ess(q)`；
- `tau_budget(h, k_min, n)`；
- `target_relative_ess(...)`；
- 基于 rESS 目标的 `solve_lambda_for_ress()`；
- tie-aware 可达目标修正；
- 全同分、并列最高分、极端尺度和有限性检查；
- `ControllerStep` 审计字段：`h`、`tau_budget`、`progress_prev`、`tau_target`、`tau_feasible`、`relative_ess`、`k_eff`、`kl_actual`、`lam`、`q`、`saturated` 等。

旧的 Shannon entropy、KL solver 和 `gamma` 参数仍保留为兼容/诊断接口，但 Full 主路径通过 rESS 求解。

### 5. 延迟 Gamma 进展反馈已实现

新增 `eureka/smc/progress.py`，实现了冻结的进展定义：

1. 根据本轮 `q` 计算 `K_eff = 1 / sum(q**2)`；
2. 取 `k = clip(ceil(K_eff), 1, M)`；
3. 从实际获得资源的父代槽位 `Z_t` 按原始 `J` 取 `P_eff`；
4. 从接受前原始子代集合 `Y_t` 按原始 `J` 取 `Y_eff`；
5. pairwise 比较中平局按 0.5 计分；
6. `Gamma_t = 2 * A_eff - 1`。

`island.py` 中首轮使用 `progress_prev=0`，当前轮计算出的 Gamma 延迟到下一轮的 `tau_target`，没有即时反馈污染当前轮。

### 6. 原始 J 分数路径已统一

`search_score` 使用原始任务分数 `J`，未引入 reward normalization。当前控制器父代选择、sigmoid 接受、best 诊断和 Fake 集成测试均围绕原始分数运行。

无效候选使用 `search_score=None`，没有用有限失败分数污染控制器。

### 7. 初始化 traceback repair 已接入

`SMCIsland.initialize()` 已支持 RF-Agent 风格的初始候选修复：

- 初始候选批量评估；
- 无效候选记录 `init_invalid`；
- 将 traceback 反馈给 `repair_batch()`；
- 同一种子连续修复失败达到阈值后改抽全新候选；
- 成功后继续填充初始种群。

Fake 测试已覆盖：

- repair 填满 N；
- repair 禁用时的 fresh retry 路径；
- 多波初始化。

但“初始化不足 N 时必须终止且不进入 mutation”这一最终状态机语义尚未完全修正，当前 `run()` 仍允许 `len(particles) >= 1` 时进入 mutation。

### 8. RF-Agent 五操作路由已保留

已保留并通过 Fake 回归测试的 action：

- `mutation_structure`；
- `mutation_parameter`；
- `crossover`；
- `path_reasoning`；
- `different_thought`。

当前 generic 模式仍是正式默认路径；RF 模式只作为兼容和 Fake 回归能力保留。action contract 只做审计，不作为 reward 接受 gate。

### 9. Archive 基础设施已新增

`eureka/smc/archive.py` 已提供：

- `CandidateArchive`；
- reward code SHA-256 去重；
- 相同代码保留更高原始 J；
- 确定性 top-k 排序；
- `snapshot()` / `restore()` / `from_snapshot()`；
- 无效候选排除。

`eureka/smc/validate_test.py` 已提供 archive top-k → validation → test 协议：

- 从 archive 选 top-k；
- validation seed panel 之间不重复；
- validation 与 test seed 不重叠；
- validation 只选唯一冠军；
- 只有 validation 冠军进入 test；
- validation/test 成本与搜索预算逻辑分离。

入口 `eureka/eureka_smc.py` 已接入三阶段复评，并输出 `validation_test_result.json` 与 `smc_summary.npz`。不过入口当前是运行结束后从 `island._registry.values()` 临时构造 archive，尚未切换到 island 内部的 accept-only 持久 archive。

### 10. 底层 checkpoint 原子 JSON 工具已完成

`eureka/smc/checkpoint.py` 已提供：

- 版本号校验；
- JSON-only 状态保存；
- 临时文件写入；
- flush + fsync；
- `os.replace()` 原子发布；
- 损坏 JSON、错误版本和不可序列化对象检查。

当前它仍只是底层工具，尚未接入 `SMCIsland.run()` 的初始化/轮次/终态边界。

### 11. 粒子反序列化接口已新增

`eureka/smc/particle.py` 已新增：

- `EvalRecord.from_json()`；
- `RewardParticle.from_json()`。

支持：

- 嵌套 `eval` 快照；
- 旧版顶层 EvalRecord 字段兼容；
- tuple 字段恢复；
- `artifact_dir` 恢复为 `Path`；
- metadata 恢复。

该接口已经写入，但当前测试暴露出 archive 仍使用旧的私有恢复函数，且该函数缺少 `EvalRecord` 导入，导致 archive round-trip 失败。

## 二、当前已验证情况

执行命令：

```bash
PYTHONPATH=. /root/miniconda3/envs/smac_test/bin/python \
  -m pytest eureka/smc/tests -q --import-mode=importlib
```

当前结果：

```text
115 passed, 2 failed
```

失败均来自：

```text
eureka/smc/tests/test_archive.py
```

具体原因：`archive.py::_restore_particle()` 仍调用 `EvalRecord(**eval_data)`，但文件当前只导入了 `RewardParticle`，没有导入 `EvalRecord`。这也是 `RewardParticle.from_json()` 已加入但 archive 尚未复用新接口造成的遗留问题。

当前尚未完成或尚未重新验证：

- island 业务 checkpoint 中断恢复；
- 预算 ledger；
- EventLogger 恢复后的 sequence 连续性；
- accepted-only provenance 完整测试；
- 初始化不足 N 的终止行为；
- 全量 `py_compile`；
- 真实 Isaac Gym / GPU / LLM smoke。

## 三、明确待处理问题

### P0：修复当前测试回归

1. 修改 `archive.py`，删除重复的私有粒子恢复逻辑，直接调用 `RewardParticle.from_json()`；或者至少补齐必要导入并确保两处恢复语义一致。
2. 重新运行 `test_archive.py` 和全量 `eureka/smc/tests`。
3. 清理 `test_island_fake.py` 中重复定义的 `test_explicit_budget_split_requires_consistent_rounds`，避免后定义覆盖前定义。
4. 检查 `kl_controller.py` 中重复的 `__all__` 导出项，虽不影响当前功能，但应整理为单一导出列表。

### P0：实现 accepted-only provenance

当前 `_round()` proposal spec 仍提前写入：

```python
accepted_transition_parent_id=c.metadata.get("state_origin_id", c.id)
```

需要改为：

- proposal 阶段只写 `proposal_parent_id`；
- `_sigmoid_accept()` 只有接受时才设置 `child.accepted_transition_parent_id`；
- 接受时更新 child 的 `state_origin_id`；
- 拒绝、invalid、noop child 不进入 accepted chain；
- clone 不成为持久 accepted 节点。

同时需要增加测试覆盖：

- rejected child 的 accepted parent 为 `None`；
- accepted child 的 accepted parent 指向被替换 slot 的真实 `state_origin_id`；
- clone 不形成 accepted edge；
- invalid child 不污染 best/registry/archive。

### P0：实现唯一的 accept-only archive/registry 提交边界

当前 `_make_particles_batch()` 在评估完成后立即执行：

```python
self._consider_best(p)
self._registry[p.id] = p
```

这会使被拒 child 仍然可能：

- 成为历史 best；
- 进入 registry；
- 被 RF 历史 action 作为供体；
- 被最终 archive 收录。

需要拆分“评估完成”和“接受提交”：

- tentative child 只保留事件、评估记录和 artifact；
- accepted child 才提交 registry/archive/best；
- 初始化粒子作为初始 accepted 状态提交；
- 被拒/invalid 候选不进入 accept-only archive。

### P0：接入 island 业务 checkpoint

需要在 `SMCIsland` 内实现可 JSON 化 snapshot/restore，至少保存：

- phase；
- 固定 N 槽位 population；
- registry；
- accept-only archive；
- best_id；
- ledger；
- round_idx；
- `budget_remaining`；
- `progress_prev`；
- `last_lambda`；
- `_id_counter`；
- island RNG state；
- proposer/evaluator 计数和 cache metadata；
- config fingerprint；
- EventLogger next sequence。

保存边界应为：

1. 完整初始化成功后；
2. 每个 mutation stage 完整结束后；
3. 搜索终态。

不应在 proposal/evaluation/acceptance 半完成时保存。

恢复时必须验证 N、M、预算拆分、seed、action 配置等契约，不重跑 init，不重复扣预算，并从最近完成的 stage 继续。

### P1：实现预算账本

目前没有 `_BudgetLedger` 或独立账本模块。需要至少区分：

```text
logical_init_slots
logical_mutation_slots
logical_search_total

llm_init_calls
llm_repair_calls
llm_mutation_calls

physical_rl_evals
logical_evals
cache_hits

noop_proposals
invalid_candidates
rejected_candidates
init_failures

validation_rl_evals
test_rl_evals
```

硬约束：

```text
logical_init_slots = 16
logical_mutation_slots = 64
logical_search_total = 80
```

repair/fresh retry 和重复评估等实际成本可以超过逻辑 slot，但不能悄悄挤占 mutation budget；noop、invalid、reject 不退款；validation/test 单独记账。

### P1：EventLogger 恢复序号

当前 `EventLogger.__init__()` 每次固定从 `_seq = 0` 开始。checkpoint 恢复后追加同一 JSONL 会造成重复序号。

需要增加：

```python
start_seq: int = 0
@property
 def next_seq(self) -> int: ...
```

并在 checkpoint 保存/恢复时使用。

### P1：入口 checkpoint/resume

`eureka/eureka_smc.py` 当前已经传入新的预算和 eta 配置，但仍直接执行：

```python
result = island.run()
```

需要增加 checkpoint 配置、路径和 resume 入口，并将恢复状态传给 island。结束后的 archive 也应使用 island 持久 archive，而不是 `_registry` 临时重建。

### P1：初始化严格 N 不变量

当前 `run()` 的判断是：

```python
if len(particles) < 1:
    return insufficient_valid_particles
```

应改为严格要求：

```python
if len(particles) != self.cfg.n_particles:
    return insufficient_valid_particles
```

不足 N 时不得进入 mutation 控制器。

### P2：真实 evaluator/proposer 计数与状态快照

当前入口能读取：

- `proposer.n_calls`；
- `proposer.total_prompt_tokens`；
- `proposer.total_completion_tokens`；
- `evaluator.n_evals`。

但这些没有统一写入 island ledger，也没有统一 checkpoint snapshot/restore。还需补充：

- init/repair/mutation 调用分类；
- physical RL eval 与 logical eval 区分；
- cache hit 计数；
- Fake evaluator/proposer RNG 与计数恢复。

### P2：完整搜索摘要与复评成本接线

`smc_summary.npz` 已写入基础搜索、archive、validation/test 字段，但尚未接入 ledger 的完整分项字段。应在 ledger 完成后统一输出，避免入口自行拼接多个不一致计数源。

## 四、当前未启动的内容

按阶段 A/B 纪律，当前没有启动以下昂贵或越界实验：

- Isaac Gym GPU/LLM 主搜索；
- 多岛 migration；
- 外部 baseline 对比；
- 消融矩阵；
- 多任务主实验；
- 真实 API/GPU 故障注入压力测试。

Fake/纯数值测试已经完成当前阶段 A 的代码验证；下一步应是受控真实 smoke，而不是直接启动主实验。

## 五、阶段 A 已完成的闭环

本轮已完成并通过测试：

1. `CandidateArchive` round-trip 改为复用 `RewardParticle.from_json()`；archive 恢复回归已修复。
2. 首版定义显式冻结为 `U_t=J_t`、`W_t=Uniform(N)`，正式预算口径为 `N=16, M=8, R=8, B_total=16+8×8=80`。
3. 已实现 range-based 无量纲控制器：`s_t`、`U_tilde`、`alpha_t`、派生 `lambda_equivalent`，父代选择和接受核均在 alpha 坐标工作。
4. `CandidateArchive`（所有 valid/finite 评估候选）与 `_registry`（所有评估节点、谱系/RF 历史）职责已分离；accepted transition 仅在真正替换 population slot 时写入。
5. 已实现 `BudgetLedger`：逻辑 init/mutation slot、候选结果、LLM/tokens、实际 RL eval/cache hit 和 validation/test 成本域独立记账。
6. `EventLogger` 支持恢复连续 sequence；`SMCIsland` 已支持 stage-boundary runtime snapshot/restore；入口支持原子 checkpoint 与 `resume_from`。
7. Fake stage-boundary 中断恢复等价测试覆盖 best、population、archive、registry、budget、ledger 与 event sequence。
8. 入口在构造 evaluator 前强制 search/validation/test 三组 seed 两两不交，并生成 `run_manifest.json` 固化任务、配置、预算、seed、checkpoint 与 Git 身份。

最新代码验证：

```bash
PYTHONPATH=. /root/miniconda3/envs/smac_test/bin/python \
  -m pytest eureka/smc/tests -q --import-mode=importlib -p no:cacheprovider
# 125 passed

/root/miniconda3/envs/smac_test/bin/python -m compileall -q eureka
# passed
```

## 六、推荐下一步执行顺序

1. 运行一个单任务、单 search/validation/test seed、小训练步数的受控真实 Isaac Gym + LLM smoke；验证 manifest、ledger、checkpoint/resume、archive 和 validation/test 产物端到端一致。
2. 为真实 proposer/evaluator 添加 adapter contract tests：API timeout/429/5xx/格式错误，训练超时/非零退出/无指标/部分输出和资源清理。
3. 冻结可执行实验矩阵：任务、方法、C0/C1/C2/C3/C5/C6 消融、Eureka baseline、预算、seed panels、主指标和 kill rule。
4. 在冻结矩阵后执行多任务、多外层 seed 的正式比较、统计汇总和预算审计；不得使用 test 结果反向调整方法定义。

## 七、当前结论

SMC-Eureka 的**阶段 A 核心框架已落地并已通过 125 项 SMC 单元/集成测试**：算法定义、无量纲 alpha 控制、预算/进展分配、archive/accepted lineage、预算账本、stage-boundary checkpoint/resume、三阶段 seed 隔离和 run manifest 已具备。

它现在可以进入受控真实 smoke，但尚不能把“代码闭环完成”等同于“论文实验结论成立”。正式主实验仍需要真实适配器端到端验证、冻结实验矩阵、baseline/消融、多任务多 seed 结果和预先定义的统计报告。
