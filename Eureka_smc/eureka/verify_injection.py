"""注入生效验证：确认修复 ISAAC_ROOT_DIR 后，SMC 注入的 reward 真正被训练使用。

注入两个不同的常数 reward(0.123 / 0.456)——它们与 actions 值无关，恒等于常数，
因此训练日志里的 gpt_reward 均值应精确等于该常数。若两次 gpt_reward 分别≈两个常数，
则证明 output_file 写入与 train.py 导入指向同一个 isaacgymenvs，注入确实生效。

从 eureka/ 目录运行：python verify_injection.py env=franka_cabinet
"""

import logging
import os
from pathlib import Path

import hydra
import openai

from utils.extract_task_code import file_to_string
from utils.create_task import create_task
from smc.evaluator import IsaacGymEvalConfig, IsaacGymEvaluator
from smc.score import ScoreConfig
from eureka_smc import EUREKA_ROOT_DIR, ISAAC_ROOT_DIR

CONST_A, CONST_B = 0.123, 0.456
VERIFY_ITERS = 50  # 只看 reward 均值，不需要学习，训练步数取小


@hydra.main(config_path="cfg", config_name="config_smc", version_base="1.1")
def main(cfg):
    logging.info(f"ISAAC_ROOT_DIR = {ISAAC_ROOT_DIR}")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    task = cfg.env.task
    suffix = cfg.suffix
    env_name = cfg.env.env_name.lower()

    # 定位 obs 源码
    env_parent = None
    for cand in ("isaac", "bidex", "dexterity"):
        d = f"{EUREKA_ROOT_DIR}/envs/{cand}"
        if os.path.isdir(d) and f"{env_name}.py" in os.listdir(d):
            env_parent = cand
            break
    task_code_string = file_to_string(f"{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py")
    output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"

    pd = f"{EUREKA_ROOT_DIR}/utils/prompts"
    prompts = {k: file_to_string(f"{pd}/{k}.txt") for k in
               ("code_output_tip", "code_feedback", "policy_feedback", "execution_error_feedback")}

    task_code_string = task_code_string.replace(task, task + suffix)
    create_task(ISAAC_ROOT_DIR, task, cfg.env.env_name, suffix)

    algo = cfg.algo
    score_cfg = ScoreConfig(metric=algo.score.metric, fallback_metric=algo.score.fallback_metric,
                            aggregate=algo.score.aggregate, window_frac=algo.score.window_frac,
                            lower=algo.score.lower, upper=algo.score.upper)
    eval_cfg = IsaacGymEvalConfig(
        isaac_root_dir=ISAAC_ROOT_DIR, eureka_root_dir=EUREKA_ROOT_DIR, task=task,
        suffix=suffix, env_name=env_name, task_code_string=task_code_string,
        output_file=output_file, max_iterations=VERIFY_ITERS, use_wandb=False)
    evaluator = IsaacGymEvaluator(score_cfg, seeds=[0], prompts=prompts, eval_cfg=eval_cfg, cache=False)

    results = {}
    for const in (CONST_A, CONST_B):
        code = (
            "def compute_reward(actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:\n"
            f"    reward = torch.zeros(actions.shape[0], device=actions.device) + {const}\n"
            '    return reward, {"dummy": reward}\n'
        )
        rec = evaluator.evaluate(code, f"verify_{const}", Path.cwd() / f"verify_{const}")
        gpt = rec.raw_metrics.get("gpt_reward_mean")
        results[const] = (rec.valid, gpt)
        logging.info(f"注入常数 {const} -> valid={rec.valid}, gpt_reward_mean={gpt}")

    print("\n===== 注入生效验证结果 =====")
    ok = True
    for const, (valid, gpt) in results.items():
        match = valid and gpt is not None and abs(gpt - const) < 0.02
        ok &= match
        print(f"  注入 {const}: valid={valid}, gpt_reward={gpt}  -> {'✓匹配' if match else '✗不匹配'}")
    print(f"\n结论：{'注入生效 ✓（bug 已修复）' if ok else '注入未生效 ✗（仍有问题）'}")


if __name__ == "__main__":
    main()
