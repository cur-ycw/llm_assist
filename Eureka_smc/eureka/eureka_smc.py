"""SMCEvolve-Eureka 单岛搜索入口（迁移计划 §11 首个可交付版本）。

在保留 Eureka 的环境代码注入 / RL 训练 / TensorBoard 反馈 / 最终多次评估能力的前提下，
把原 best-of-N 单谱系循环替换为单岛序贯蒙特卡洛搜索：ESS 自适应温度、系统重采样、单一
eureka_reflection proposal、reward-only MH 接受、到达终端桥分布或预算上限停止，最后用与
搜索 panel 不重叠的 held-out seeds 复评。

从 ``eureka/`` 目录运行：``python eureka_smc.py env=cartpole``（cwd 需含 smc/ 与 utils/）。
不改动原 ``eureka.py`` 的 best-of-N 行为。
"""

import logging
import os
from pathlib import Path

import hydra
import numpy as np
import openai

from utils.extract_task_code import file_to_string
from utils.create_task import create_task

from smc.actions import ActionConfig
from smc.evaluator import IsaacGymEvalConfig, IsaacGymEvaluator
from smc.event_logger import EventLogger
from smc.island import SMCIsland, SMCIslandConfig
from smc.proposer import EurekaReflectionProposer, TaskContext
from smc.score import ScoreConfig

EUREKA_ROOT_DIR = os.getcwd()


def _resolve_isaac_root() -> str:
    """指向 conda env 实际安装的 isaacgymenvs 包目录。

    关键正确性修复：训练子进程 import 的是已安装的 isaacgymenvs（本机为
    /root/ycw/Eureka/isaacgymenvs），若 output_file 写到 Eureka_smc 自带的
    isaacgymenvs，则 reward 注入写读不在同一处、注入根本不生效。用 find_spec
    定位已安装包，保证写=读。找不到时回退到仓库自带路径。
    """
    import importlib.util
    spec = importlib.util.find_spec("isaacgymenvs")
    if spec is not None and spec.origin:
        return os.path.dirname(os.path.abspath(spec.origin))
    return f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"


ISAAC_ROOT_DIR = _resolve_isaac_root()


@hydra.main(config_path="cfg", config_name="config_smc", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.chatanywhere.tech/v1")

    task = cfg.env.task
    suffix = cfg.suffix
    env_name = cfg.env.env_name.lower()
    logging.info(f"SMC-Eureka | task={task} | model={cfg.model}")

    # ---- 任务代码与 obs ----
    # 本仓库 envs/ 为 isaac(单环境) + bidex(双手 ShadowHand)，官方 eureka.py 硬编码的
    # dexterity 目录在此不存在，故遍历候选目录定位任务 obs 源码。
    env_parent = None
    for cand in ("isaac", "bidex", "dexterity"):
        d = f"{EUREKA_ROOT_DIR}/envs/{cand}"
        if os.path.isdir(d) and f"{env_name}.py" in os.listdir(d):
            env_parent = cand
            break
    if env_parent is None:
        raise FileNotFoundError(f"找不到任务 obs 源码 envs/*/{env_name}.py")
    task_file = f"{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py"
    task_obs_file = f"{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py"
    task_code_string = file_to_string(task_file)
    task_obs_code_string = file_to_string(task_obs_file)
    output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"

    # ---- prompts ----
    pd = f"{EUREKA_ROOT_DIR}/utils/prompts"
    prompts = {k: file_to_string(f"{pd}/{k}.txt") for k in
               ("initial_system", "code_output_tip", "code_feedback", "initial_user",
                "reward_signature", "policy_feedback", "execution_error_feedback",
                "initial_failed_feedback")}
    initial_system = prompts["initial_system"].format(
        task_reward_signature_string=prompts["reward_signature"]) + prompts["code_output_tip"]
    initial_user = prompts["initial_user"].format(
        task_obs_code_string=task_obs_code_string, task_description=cfg.env.description)

    task_code_string = task_code_string.replace(task, task + suffix)
    create_task(ISAAC_ROOT_DIR, task, cfg.env.env_name, suffix)

    # ---- 装配 SMC 组件 ----
    algo = cfg.algo
    # search_score 的归一化上/下界是**任务相关**的（consecutive_successes 在 FrankaCabinet
    # 是成功 env 占比 ∈[0,1] → upper=1；在手部任务是连续达标计数 → upper≈50；gt_reward 回退
    # 任务是回合回报 → 上千甚至为负）。故优先取 env 配置里的 score_upper/score_lower，
    # 缺省才用 algo.score 的全局 fallback。metric 同理可按任务覆盖。
    score_metric = cfg.env.get("score_metric", algo.score.metric)
    score_lower = cfg.env.get("score_lower", algo.score.lower)
    score_upper = cfg.env.get("score_upper", algo.score.upper)
    logging.info(f"Score: metric={score_metric}, lower={score_lower}, upper={score_upper} "
                 f"(env override: {'score_upper' in cfg.env})")
    score_cfg = ScoreConfig(
        metric=score_metric, fallback_metric=algo.score.fallback_metric,
        aggregate=algo.score.aggregate, window_frac=algo.score.window_frac,
        lower=score_lower, upper=score_upper)

    eval_cfg = IsaacGymEvalConfig(
        isaac_root_dir=ISAAC_ROOT_DIR, eureka_root_dir=EUREKA_ROOT_DIR, task=task,
        suffix=suffix, env_name=env_name, task_code_string=task_code_string,
        output_file=output_file, max_iterations=cfg.policy_train_iterations,
        use_wandb=cfg.use_wandb, wandb_username=cfg.wandb_username,
        wandb_project=cfg.wandb_project, capture_video=cfg.capture_video)

    ctx = TaskContext(initial_system=initial_system, initial_user=initial_user,
                      code_output_tip=prompts["code_output_tip"], model=cfg.model,
                      temperature=cfg.temperature)

    # ---- RF-Agent 五操作路由（Phase-3a 变异 m1/m2；Phase-3b 历史 crossover/path/different）----
    actions_cfg = algo.get("actions", None)
    action_cfg = None
    action_prompts: dict = {}
    if actions_cfg is not None and actions_cfg.get("mode", "generic") == "rf":
        enabled = list(actions_cfg.get("enabled", ["mutation_structure", "mutation_parameter"]))
        action_cfg = ActionConfig(
            mode="rf", enabled=enabled,
            rf_ratio=list(actions_cfg.get("rf_ratio", [2, 2, 2, 1, 1])),
            gate_contracts=bool(actions_cfg.get("gate_contracts", False)),
            crossover_k=int(actions_cfg.get("crossover_k", 2)),
            history_k=int(actions_cfg.get("history_k", 4)))
        # 只加载已启用 action 的指令模板（rf_actions/<name>.txt）
        for name in enabled:
            action_prompts[name] = file_to_string(f"{pd}/rf_actions/{name}.txt")
        logging.info(f"Actions: mode=rf, enabled={enabled}, "
                     f"weights={action_cfg.enabled_weights()}, gate={action_cfg.gate_contracts}, "
                     f"crossover_k={action_cfg.crossover_k}, history_k={action_cfg.history_k}")
    else:
        logging.info("Actions: mode=generic (单一 eureka_reflection)")

    proposer = EurekaReflectionProposer(ctx, openai, action_prompts=action_prompts,
                                        initial_failed_prompt=prompts["initial_failed_feedback"])
    max_concurrent = algo.evaluation.get("max_concurrent_evals", 6)
    evaluator = IsaacGymEvaluator(
        score_cfg, seeds=list(algo.evaluation.search_seed_panel),
        prompts=prompts, eval_cfg=eval_cfg, cache=algo.evaluation.cache,
        max_concurrent=max_concurrent)

    island = SMCIsland(
        SMCIslandConfig(
            island_id=0, n_particles=algo.n_particles,
            children_per_round=algo.children_per_round, budget=algo.budget,
            k_min=algo.k_min, gamma=algo.gamma,
            max_init_repair=algo.max_init_repair, max_same_repair=algo.max_same_repair,
            seed=algo.seed, action_cfg=action_cfg),
        proposer, evaluator,
        EventLogger(workspace_dir / "smc_events.jsonl"),
        artifact_root=workspace_dir / "candidates")

    result = island.run()
    logging.info(f"SMC done: reason={result.termination_reason}, rounds={result.n_rounds}, "
                 f"budget_used={result.budget_used}, last_lambda={result.last_lambda:.4f}, "
                 f"best_score={result.best.search_score if result.best else None}")
    logging.info(f"LLM calls={proposer.n_calls}, prompt_tok={proposer.total_prompt_tokens}, "
                 f"completion_tok={proposer.total_completion_tokens}, RL evals={evaluator.n_evals}")

    if result.best is None:
        logging.info("无有效最优候选，终止（请检查 candidates/ 下的训练日志）。")
        return

    best_code_path = workspace_dir / "best_reward_code.py"
    best_code_path.write_text(result.best.reward_code)
    logging.info(f"Best reward code saved to {best_code_path}")

    # ---- held-out 复评（与搜索 panel 不重叠的 seeds，计划 §5.2）----
    heldout = list(algo.evaluation.heldout_seed_panel)
    logging.info(f"Held-out re-evaluation on seeds {heldout}")
    heldout_eval = IsaacGymEvaluator(score_cfg, seeds=heldout, prompts=prompts,
                                     eval_cfg=eval_cfg, cache=False,
                                     max_concurrent=max_concurrent)
    rec = heldout_eval.evaluate(result.best.reward_code, "heldout",
                                workspace_dir / "heldout")
    per_seed = rec.reward_components.get("per_seed", [])
    logging.info(f"Held-out raw search_score (J): mean={rec.search_score}, "
                 f"per_seed={per_seed}")
    np.savez(workspace_dir / "smc_summary.npz",
             termination_reason=result.termination_reason, n_rounds=result.n_rounds,
             budget_used=result.budget_used, last_lambda=result.last_lambda,
             best_search_score=result.best.search_score,
             heldout_search_score=rec.search_score if rec.search_score is not None else np.nan,
             heldout_per_seed=np.array(per_seed, dtype=float),
             llm_calls=proposer.n_calls, rl_evals=evaluator.n_evals)


if __name__ == "__main__":
    main()
