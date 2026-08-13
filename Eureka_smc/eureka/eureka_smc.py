"""SMCEvolve-Eureka 单岛搜索入口（迁移计划 §11 首个可交付版本）。

在保留 Eureka 的环境代码注入 / RL 训练 / TensorBoard 反馈 / 最终多次评估能力的前提下，
把原 best-of-N 单谱系循环替换为单岛序贯蒙特卡洛搜索：ESS 自适应温度、系统重采样、单一
eureka_reflection proposal、reward-only MH 接受、到达终端桥分布或预算上限停止，最后用与
搜索 panel 不重叠的 held-out seeds 复评。

从 ``eureka/`` 目录运行：``python eureka_smc.py env=cartpole``（cwd 需含 smc/ 与 utils/）。
不改动原 ``eureka.py`` 的 best-of-N 行为。
"""

import json
import logging
import os
from pathlib import Path

import hydra
import numpy as np
import openai
from omegaconf import OmegaConf

from utils.extract_task_code import file_to_string
from utils.create_task import create_task

from smc.actions import ActionConfig
from smc.evaluator import IsaacGymEvalConfig, IsaacGymEvaluator
from smc.checkpoint import load_checkpoint, save_checkpoint
from smc.event_logger import EventLogger
from smc.island import SMCIsland, SMCIslandConfig
from smc.proposer import EurekaReflectionProposer, TaskContext
from smc.run_manifest import write_run_manifest
from smc.score import ScoreConfig
from smc.validate_test import check_seed_panels, run_validation_test

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

    # 迭代数（RF-Agent 每任务标准）：cfg/env/<task>.yaml 为 per-task 真源；顶层
    # policy_train_iterations / test_policy_train_iterations 为 CLI 覆盖钩子（非 null 时优先，供 smoke）。
    # RF-Agent 多数任务 test = search 的 2×（search 减半以加速搜索，test 保基准全量）。
    search_iters = cfg.get("policy_train_iterations")
    if search_iters is None:
        search_iters = cfg.env.get("policy_train_iterations", None)
    search_iters = int(search_iters) if search_iters is not None else 3000
    test_iters = cfg.get("test_policy_train_iterations")
    if test_iters is None:
        test_iters = cfg.env.get("test_policy_train_iterations", None)
    test_iters = int(test_iters) if test_iters is not None else search_iters
    logging.info(f"Iterations (RF-Agent standard): search={search_iters}, test={test_iters}")

    eval_cfg = IsaacGymEvalConfig(
        isaac_root_dir=ISAAC_ROOT_DIR, eureka_root_dir=EUREKA_ROOT_DIR, task=task,
        suffix=suffix, env_name=env_name, task_code_string=task_code_string,
        output_file=output_file, max_iterations=search_iters,
        use_wandb=cfg.use_wandb, wandb_username=cfg.wandb_username,
        wandb_project=cfg.wandb_project, capture_video=cfg.capture_video,
        startup_timeout_seconds=algo.evaluation.get("startup_timeout_seconds", 600.0),
        training_timeout_seconds=algo.evaluation.get("training_timeout_seconds", None),
        lock_timeout_seconds=algo.evaluation.get("lock_timeout_seconds", 300.0))
    # test/held-out 复评用独立迭代数（RF-Agent test_max_iterations），其余参数与 search 一致。
    # 所有字段均为不可变标量，copy.copy 安全；_config_salt 含 max_iterations，故 test 缓存与 search 天然分离。
    import copy
    test_eval_cfg = copy.copy(eval_cfg)
    test_eval_cfg.max_iterations = test_iters

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
    search_seeds = list(algo.evaluation.search_seed_panel)
    reeval = algo.get("reevaluation", None)
    reeval_enabled = reeval is not None and reeval.get("enabled", True)
    validation_enabled = reeval_enabled and reeval.get("validation_enabled", True)
    if reeval_enabled:
        # validation 关闭时无中间复评层，validation_seeds 记为空（不占预算、不参与不相交检查）。
        validation_seeds = list(reeval.validation_seed_panel) if validation_enabled else []
        test_seeds = list(reeval.test_seed_panel)
    else:
        validation_seeds = list(algo.evaluation.validation_seed_panel)
        test_seeds = list(algo.evaluation.test_seed_panel)
    check_seed_panels(validation_seeds, test_seeds, search_seeds)

    # ---- 冠军 test-only 重评（retest_champion_code=<path>）----
    # 给定已保存的冠军奖励代码路径时，跳过整段搜索，仅用当前评估器对该冠军在 test seeds 上
    # 复评（cache=false）。用于在评估器口径变更（如去掉训练超时）后，对搜索阶段不受影响、
    # 仅最终 test 被削的冠军做干净补跑，无需重跑 ~24h 的搜索。
    retest_path = cfg.get("retest_champion_code", None)
    if retest_path:
        champion_code = Path(retest_path).read_text()
        logging.info(f"[RETEST] champion={retest_path} | test_seeds={test_seeds} | "
                     f"test_iters={test_iters} | training_timeout="
                     f"{test_eval_cfg.training_timeout_seconds}")
        retest_eval = IsaacGymEvaluator(score_cfg, seeds=test_seeds, prompts=prompts,
                                        eval_cfg=test_eval_cfg, cache=False,
                                        max_concurrent=max_concurrent)
        rec = retest_eval.evaluate(champion_code, "champion_retest",
                                   workspace_dir / "retest")
        per_seed = rec.reward_components.get("per_seed", []) if rec else []
        logging.info(f"[RETEST] champion test raw J mean={rec.search_score}, per_seed={per_seed}")
        (workspace_dir / "retest_result.json").write_text(json.dumps(
            {"champion_code": str(retest_path), "test_seeds": test_seeds,
             "test_iters": test_iters, "training_timeout_seconds": test_eval_cfg.training_timeout_seconds,
             "test_search_score": rec.search_score, "per_seed": per_seed,
             "valid": rec.valid, "error": rec.error},
            default=str, ensure_ascii=False, indent=2))
        return

    manifest_budget = {
        "n_particles": int(algo.n_particles),
        "init_budget": int(algo.init_budget),
        "children_per_round": int(algo.children_per_round),
        "mutation_rounds": int(algo.mutation_rounds),
        "mutation_budget": int(algo.mutation_budget),
        "budget_total": int(algo.budget),
    }
    write_run_manifest(
        workspace_dir / "run_manifest.json",
        resolved_config=OmegaConf.to_container(cfg, resolve=True),
        task=cfg.env.task,
        env_name=cfg.env.env_name,
        search_seeds=search_seeds,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
        budget=manifest_budget,
        checkpoint=dict(algo.get("checkpoint", {})),
        project_root=Path(__file__).resolve().parents[1],
    )
    evaluator = IsaacGymEvaluator(
        score_cfg, seeds=list(algo.evaluation.search_seed_panel),
        prompts=prompts, eval_cfg=eval_cfg, cache=algo.evaluation.cache,
        max_concurrent=max_concurrent)

    checkpoint_cfg = algo.get("checkpoint", {})
    checkpoint_path = Path(checkpoint_cfg.path) if checkpoint_cfg.get("enabled", False) else None
    resume_from = checkpoint_cfg.get("resume_from", None)
    resume_state = load_checkpoint(resume_from) if resume_from else None
    event_start_seq = int(resume_state.get("event_next_seq", 0)) if resume_state else 0
    event_log = EventLogger(workspace_dir / "smc_events.jsonl", start_seq=event_start_seq)

    island = SMCIsland(
        SMCIslandConfig(
            island_id=0, n_particles=algo.n_particles,
            children_per_round=algo.children_per_round, mutation_rounds=algo.get("mutation_rounds"),
            budget=algo.budget, init_budget=algo.get("init_budget"),
            mutation_budget=algo.get("mutation_budget"),
            k_min=algo.k_min, gamma=algo.gamma, eta=algo.get("eta", 1.0),
            max_init_repair=algo.max_init_repair, max_same_repair=algo.max_same_repair,
            max_mutation_repair=algo.get("max_mutation_repair", 0),
            seed=algo.seed, action_cfg=action_cfg,
            accept_sharpness=float(algo.get("accept_sharpness", 1.0))),
        proposer, evaluator,
        event_log,
        artifact_root=workspace_dir / "candidates")

    checkpoint_callback = (
        (lambda state: save_checkpoint(checkpoint_path, state))
        if checkpoint_path is not None else None
    )
    result = island.run(resume_state=resume_state, checkpoint_callback=checkpoint_callback)
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

    # ---- archive → validation → test 三阶段复评（计划 §5 阶段 D）----
    # archive 汇集所有已完成有效评估的候选（按代码 hash 去重、保留最高原始 J），
    # 与 island 的 best/population 分离。validation/test seed panel 与 search 及彼此
    # 均不重叠，成本单独记账、不计入搜索预算 B。
    reeval = algo.get("reevaluation", None)
    archive = island.archive
    logging.info(f"Archive size={len(archive)} (deduped valid candidates)")

    if reeval is not None and reeval.get("enabled", True) and len(archive) > 0:
        validation_enabled = reeval.get("validation_enabled", True)
        test_seeds = list(reeval.test_seed_panel)
        test_eval = IsaacGymEvaluator(score_cfg, seeds=test_seeds, prompts=prompts,
                                      eval_cfg=test_eval_cfg, cache=False,
                                      max_concurrent=max_concurrent)
        if validation_enabled:
            top_k = int(reeval.get("archive_top_k", 3))
            val_seeds = list(reeval.validation_seed_panel)
            logging.info(f"Reevaluation: top_k={top_k}, validation={val_seeds}, test={test_seeds}")
            validation_eval = IsaacGymEvaluator(score_cfg, seeds=val_seeds, prompts=prompts,
                                                eval_cfg=test_eval_cfg, cache=False,
                                                max_concurrent=max_concurrent)
            vt = run_validation_test(
                archive, validation_eval, test_eval, top_k=top_k,
                validation_seeds=val_seeds, test_seeds=test_seeds,
                artifact_root=workspace_dir / "reeval")
            validation_rl_evals = validation_eval.n_evals
        else:
            # validation 关闭：冠军直接取 archive search_score top-1，只做 test held-out。
            logging.info(f"Reevaluation: validation disabled, "
                         f"champion=archive search_score top-1, test={test_seeds}")
            vt = run_validation_test(
                archive, None, test_eval, top_k=1,
                validation_seeds=[], test_seeds=test_seeds,
                artifact_root=workspace_dir / "reeval")
            validation_rl_evals = 0
        test_rec = vt.test_record
        test_per_seed = test_rec.reward_components.get("per_seed", []) if test_rec else []
        logging.info(f"Champion={vt.selected_candidate_id}; "
                     f"test raw J mean={test_rec.search_score if test_rec else None}, "
                     f"per_seed={test_per_seed}")
        (workspace_dir / "validation_test_result.json").write_text(
            json.dumps(vt.to_json(), default=str, ensure_ascii=False, indent=2))
        # validation/test 成本单独记账，不并入搜索 evaluator.n_evals
        np.savez(
            workspace_dir / "smc_summary.npz",
            termination_reason=result.termination_reason, n_rounds=result.n_rounds,
            budget_used=result.budget_used, last_lambda=result.last_lambda,
            best_search_score=result.best.search_score, archive_size=len(archive),
            selected_candidate_id=vt.selected_candidate_id or "",
            test_search_score=test_rec.search_score if (test_rec and test_rec.search_score is not None) else np.nan,
            test_per_seed=np.array(test_per_seed, dtype=float),
            llm_calls=proposer.n_calls, rl_evals=evaluator.n_evals,
            validation_rl_evals=validation_rl_evals, test_rl_evals=test_eval.n_evals)
    else:
        # 兼容旧路径：单一 best 的 held-out 复评（仅在 reevaluation 关闭时使用）。
        heldout = list(algo.evaluation.heldout_seed_panel)
        logging.info(f"Reevaluation disabled; legacy single-best held-out on {heldout}")
        heldout_eval = IsaacGymEvaluator(score_cfg, seeds=heldout, prompts=prompts,
                                         eval_cfg=test_eval_cfg, cache=False,
                                         max_concurrent=max_concurrent)
        rec = heldout_eval.evaluate(result.best.reward_code, "heldout",
                                    workspace_dir / "heldout")
        per_seed = rec.reward_components.get("per_seed", [])
        logging.info(f"Held-out raw search_score (J): mean={rec.search_score}, per_seed={per_seed}")
        np.savez(workspace_dir / "smc_summary.npz",
                 termination_reason=result.termination_reason, n_rounds=result.n_rounds,
                 budget_used=result.budget_used, last_lambda=result.last_lambda,
                 best_search_score=result.best.search_score,
                 heldout_search_score=rec.search_score if rec.search_score is not None else np.nan,
                 heldout_per_seed=np.array(per_seed, dtype=float),
                 llm_calls=proposer.n_calls, rl_evals=evaluator.n_evals)


if __name__ == "__main__":
    main()
