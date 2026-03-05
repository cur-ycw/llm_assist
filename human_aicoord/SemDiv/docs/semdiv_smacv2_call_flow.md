# SemDiv 里如何调用 SMACv2

## 1. 运行方式

在 `language/` 下执行：

```bash
cd human_aicoord/SemDiv/language
python semdiv.py
```

主程序里默认 `env = 'sc2'`（约 994 行），会走 sc2 分支。

---

## 2. sc2 时用到的 SMACv2 相关路径

| 作用 | 路径 |
|------|------|
| 初始/备份 | `pymarl/src/envs/smacv2/smacv2/env/starcraft2/additional_reward_init.py` |
| LLM 写奖励逻辑 | `pymarl/src/envs/smacv2/smacv2/env/starcraft2/additional_reward_1.py` |

SemDiv 不直接「跑」环境，而是：

- 用 LLM 在 `additional_reward_1.py` 里写 `llm1`, `llm2`, ... 等函数；
- 然后通过 **shell 脚本调 pymarl**，由 pymarl 里 **sc2_v2** 环境去读这些 reward 并跑 SMACv2。

---

## 3. 调用 SMACv2 的完整链

```
semdiv.py (env='sc2')
  │
  ├─ 用 additional_reward_init.py 覆盖 additional_reward_1.py
  ├─ SemDiv(env, env_file, timing), env_file = additional_reward_1.py
  └─ semdiv.run()
        │
        ├─ generate_behavior() / LLM 写代码到 env_file (additional_reward_1.py)
        ├─ train_tm()  →  bash pymarl/src/scripts/semdiv_train_tm_sp_llm.sh ...
        │                    │
        │                    └─ cd ../pymarl
        │                       python src/main.py --env-config=sc2_v2 ...
        │                         → pymarl 里 REGISTRY["sc2_v2"] = StarCraft2Env2Wrapper
        │                         → 即 pymarl/src/envs/sc2_v2_wrapper.py
        │                         → 内部用 smacv2 的 StarCraft2Env + additional_reward_1 里的 llm*
        │
        ├─ train_ego() → bash pymarl/src/scripts/semdiv_train_ego.sh ...  (同样 --env-config=sc2_v2)
        └─ 评估/渲染 等也通过脚本调 pymarl，--env-config=sc2_v2
```

也就是说：**真正「跑」SMACv2 的是 pymarl**，semdiv 只负责：

- 设定 `env='sc2'`；
- 维护 `additional_reward_1.py`（LLM 写的 reward）；
- 通过 **bash 脚本** 调 pymarl 的 `main.py --env-config=sc2_v2`。

---

## 4. 关键脚本与参数（sc2）

- **TM 训练**：`pymarl/src/scripts/semdiv_train_tm_sp_llm.sh`  
  - 执行：`python src/main.py --sp --config=vdn --env-config=sc2_v2 ... env_args.env_id=... env_args.additional_reward_id=$llm_tm_index`
- **Ego 训练**：`pymarl/src/scripts/semdiv_train_ego.sh`  
  - 同样在 pymarl 里跑，`--env-config=sc2_v2`。
- **结果目录**：`pymarl/results/pymarl/sc2_v2/<run_name>/`（见 `check_status_pymarl` 里 `pymarl_env = 'sc2_v2'`）。

---

## 5. 小结

- **SemDiv 里「能跑 sc2」** = 用 pymarl 的 **sc2_v2** 配置（即 SMACv2）。
- **SMACv2 的入口**在 pymarl 里：`envs/__init__.py` 的 `REGISTRY["sc2_v2"] = StarCraft2Env2Wrapper`，底层是 `smacv2` 的 `StarCraft2Env`。
- SemDiv 通过 **bash 脚本 + `--env-config=sc2_v2`** 调用上述环境；**没有**在 semdiv.py 里直接 import 或 new 一个 SMACv2 环境实例。
- 游戏本体（StarCraft II）由 pymarl 在运行 `main.py` 时按 `SC2PATH` / 默认路径查找；和 HARL 用的是同一套 pymarl + 同一套 SC2 路径逻辑。
