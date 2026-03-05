# SMAC 场景下 MAPPO Self-Play 训练框架设计

## 一、现状与目标

- **当前 SMAC 实现**：PyMARL + VDN（off-policy），`n_agents=2`，`n_ego=1`；TM 用 `--sp` 训、Ego 用 `--continual` + `episode_xp` 与多个 TM 对战。
- **目标**：在 SMAC（如 10gen_terran，2 己方 vs 4 敌方）上设计一套 **MAPPO（on-policy）+ self-play** 的训练框架，可与现有 SemDiv 流程对接或独立使用。

---

## 二、MAPPO 在仓库中的已有支持

- **HARL**（`human_aicoord/SemDiv/HARL`）已提供：
  - **MAPPO**：`OnPolicyMARunner`，centralized critic（EP/FP），可选 `share_param`（所有 agent 共用一个 actor）。
  - **MAPPO FCP**（Fix / Continual / Population）：`OnPolicyMARunnerFCP` + `train_ego.py`，用于 **Ego vs 固定队友种群**：
    - `load_agent_id`：固定队友的 agent id（如 1）。
    - `population_model_dirs`：多个 TM checkpoint 目录，每步 rollout 时从 `np.random.choice(self.population)` 采样一个作为队友策略。
    - 只训练 Ego（另一 agent），队友不更新。
- **环境**：`envs_tools.py` 中已支持 `smac` / `smacv2`；若本仓库未包含 smacv2 实现，需对接 PyMARL 侧的 `sc2_v2` 或自建 smacv2 的 obs/state/action 接口以符合 HARL 的 VecEnv 约定。

---

## 三、Self-Play 的两种用法

### 3.1 方式 A：全队同策略 Self-Play（训练“基础/TM”策略）

- **含义**：所有己方单位共用同一套 MAPPO 策略，在固定 SMAC 地图上合作打环境，等价于“自己和自己协作”的 self-play。
- **实现**：
  - 使用 HARL：`train.py --algo mappo --env smacv2 --exp_name <name>`
  - 在算法配置中设置 `share_param: True`（若 HARL 的 smacv2 支持），则 2 个 agent 共用一个 actor；否则 2 个 actor 分别训练但目标一致（团队回报）。
  - 无需对手池；训练至收敛即得到一个“TM 策略”，可存为 checkpoint 用作后续 Ego 的对手或队友。
- **用途**：得到多个不同行为（例如不同 `additional_reward_id` 或不同地图）的 TM，作为 Ego 的 population。

### 3.2 方式 B：Ego vs 种群（当前 FCP 的 SMAC 版）

- **含义**：Agent 0 = Ego（在训），Agent 1 = Teammate（固定），从 **population** 中每局或每步随机采样一个 TM 策略；Ego 学习与不同队友协作。
- **实现**：
  - **Phase 1**：用方式 A 或现有 SemDiv TM 流程，训练多个 TM（不同行为或不同 seed），得到多个 checkpoint 目录。
  - **Phase 2**：使用 HARL 的 FCP 入口：
    - `train_ego.py --algo mappo --env smacv2 --exp_name <ego_name>`
    - 传参：`--load_agent_id 1`，`--population_model_dirs "[path1,path2,...]"`，`--model_dir <last_ego_dir>`（若要做 continual）。
  - 在 FCP runner 中，`load_agent_id=1` 的 agent 每步由 `np.random.choice(self.population)` 给出动作，不更新参数；只更新 Ego（agent 0）和 centralized critic。
- **与 PyMARL 的对应**：相当于 PyMARL 的 `run_continual` + `episode_xp` + `checkpoint_path_tm_lst`，只是算法从 VDN 换成 MAPPO，接口从 PyMARL 换成 HARL。

---

## 四、SMAC 场景下的具体设计要点

### 4.1 环境与地图

- 与现有 SemDiv 对齐：使用 **sc2_v2** 配置（如 `map_name: "10gen_terran"`，`n_units: 2`，`n_enemies: 4`）。
- 在 HARL 侧需要：
  - **smacv2** 的 env 封装：提供 `observation_space`、`action_space`、`share_observation_space`（centralized state）、以及 `step` / `reset` 的批量接口（与 HARL 的 VecEnv 一致）。
  - 若 HARL 已有 smacv2，只需在 env_args 中传入与 PyMARL 一致的 `map_name`、`capability_config` 等；若没有，需要基于当前 PyMARL 的 `sc2_v2_wrapper` / `StarCraft2Env` 做一层适配（obs/state 维度、mask、reward 等）。

### 4.2 算法与网络

- **MAPPO**：
  - **Critic**：输入 global state（或 concatenated obs），输出 V(s)；SMAC 通常用 **EP**（global state）更合适。
  - **Actor**：每 agent 独立 π(a|o)（或 `share_param` 时共用），支持 RNN、action mask（avail_actions）。
- **Self-play 相关**：
  - 方式 A：无额外模块；只需 `share_param` 或独立 actor 训练团队回报。
  - 方式 B：FCP 已实现 population 采样与 fix 某 agent；只需保证 SMAC 的 checkpoint 格式与 HARL 一致（如 `actor_agent{id}_{step}.pt`）。

### 4.3 训练流程（推荐：与 SemDiv 对齐）

1. **TM 阶段（多个行为）**
   - 对每个行为 k（如不同 `additional_reward_id` 或不同 reward 设计）：
     - 用 MAPPO（方式 A）在 smacv2 上训练，得到 checkpoint 目录 `models/tm_k`。
   - 可选：沿用 SemDiv 的 LLM 生成行为 + 写 `additional_reward`，只在训练时把 PyMARL 换成 HARL MAPPO。

2. **Ego 阶段**
   - 使用 HARL FCP：`train_ego.py --algo mappo --env smacv2 ... --load_agent_id 1 --population_model_dirs "[tm_1,tm_2,...]"`。
   - Ego 为 agent 0，agent 1 从 population 中采样；训练 Ego 与 centralized critic，直到收敛或达到预定步数。

3. **评估**
   - 固定 Ego + 固定某个 TM（或轮换 TM），在 smacv2 上跑多局，记录胜率/回报/合作指标；可与现有 `semdiv_eval_xp` 的指标对齐（如 return_original、desired_ratio 等）。

### 4.4 与 PyMARL 的接口差异（若需双轨）

- **PyMARL**：VDN、episode buffer、`runner=episode_xp`、`n_ego=1`、merge_actions(ego, tm)。
- **HARL MAPPO**：on-policy buffer、EP/FP state、FCP 中 population 采样的是“整个 actor 对象”，不需要显式 merge_actions，只需在 collect 时对 `load_agent_id` 用 population 的 actor 输出动作。
- 若希望 **评估脚本统一**（例如 selection.py 或 res.json 同时支持 PyMARL 与 HARL），需要：
  - 统一 Ego/TM 的 checkpoint 路径约定；
  - 统一评估脚本的调用方式（例如 HARL 提供类似 `semdiv_eval_xp.sh` 的入口，输出相同格式的 res.json）。

---

## 五、实现检查清单（SMAC + MAPPO self-play）

- [ ] **HARL 侧**：确认 `smacv2` 在本地可运行（`make_train_env("smacv2", ...)`），且 obs/state/action 与当前 sc2_v2（10gen_terran）一致；若无，则实现或移植 smacv2_env。
- [ ] **MAPPO 配置**：为 smacv2 建一份 `mappo` + `smacv2` 的默认 yaml（episode_length、n_rollout_threads、map_name、use_recurrent_policy 等），与 tuned_configs 中 smacv2 的 happo 配置对齐。
- [ ] **方式 A**：跑通 `train.py --algo mappo --env smacv2 --exp_name sp_tm`，保存 checkpoint；确认 `share_param` 是否可用及效果。
- [ ] **方式 B**：用方式 A 得到 2～5 个 TM 目录，跑通 `train_ego.py --algo mappo --env smacv2 ... --load_agent_id 1 --population_model_dirs "[...]"`，确认 Ego 在训练、TM 从 population 采样且不更新。
- [ ] **Additional reward**：若 SMAC 要接 SemDiv 的“不同行为”，需在 smacv2 的 env 或 wrapper 中支持 `additional_reward_id`（或等价机制），与 PyMARL 的 `additional_reward_1.py` 行为一致。
- [ ] **评估与日志**：Ego vs 每个 TM 的评估脚本、TensorBoard 指标、以及是否写 res.json 与 selection.py 兼容。

---

## 六、小结

- **MAPPO self-play 在 SMAC 上**可以拆成两种形态：
  1. **全队同策略**：HARL `mappo` + `share_param`（或双 actor 共训），用于训练 TM/基础策略。
  2. **Ego vs 种群**：HARL `mappo_fcp` + `train_ego.py` + `population_model_dirs`，用于训练能与多种队友协作的 Ego。
- 仓库中已有 **MAPPO 与 FCP 的完整实现**（HARL），SMAC 侧主要工作是：**保证 smacv2 环境在 HARL 中可用**，以及 **config/脚本/评估与现有 SemDiv SMAC 流程对齐**。按上述检查清单逐项打通即可得到一套完整的 SMAC MAPPO self-play 训练框架。
