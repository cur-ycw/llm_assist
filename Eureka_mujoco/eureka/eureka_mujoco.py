"""EUREKA main loop, adapted for SB3 + gymnasium MuJoCo + MetaWorld.

Differences from the original IsaacGym eureka.py:
  - No `self.rew_buf` injection — LLM reward is written as a standalone .py
    file with a top-level `compute_reward(env, obs, action)` and passed to
    train_mujoco.py via --gpt_reward_module.
  - RL backend is `rl/train_mujoco.py` (SB3 PPO), not isaacgymenvs/train.py.
  - Task-specific observation context is provided via explicit `*_obs.py`
    files so locomotion / manipulation / geometric-control tasks do not share
    the wrong interface semantics.
"""
import hydra
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import openai
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_interfaces import get_task_interface
from utils.extract_task_code import get_function_signature
from utils.file_utils import load_tensorboard_logs
from utils.misc import block_until_training, filter_traceback, set_freest_gpu
from utils.extract_task_code import file_to_string

EUREKA_ROOT_DIR = os.getcwd()
RL_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../rl"

PROMPT_BUNDLE_BY_INTERFACE = {
    "walker": {
        "initial_system": "initial_system_walker.txt",
        "reward_signature": "reward_signature_walker.txt",
        "policy_feedback": "policy_feedback_locomotion.txt",
    },
    "ant": {
        "initial_system": "initial_system_ant.txt",
        "reward_signature": "reward_signature_ant.txt",
        "policy_feedback": "policy_feedback_locomotion.txt",
    },
    "reach": {
        "initial_system": "initial_system_reach.txt",
        "reward_signature": "reward_signature_reach.txt",
        "policy_feedback": "policy_feedback_manipulation.txt",
    },
    "pick": {
        "initial_system": "initial_system_pick.txt",
        "reward_signature": "reward_signature_pick.txt",
        "policy_feedback": "policy_feedback_manipulation.txt",
    },
    "door": {
        "initial_system": "initial_system_door.txt",
        "reward_signature": "reward_signature_door.txt",
        "policy_feedback": "policy_feedback_manipulation.txt",
    },
}


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.chatanywhere.tech/v1")

    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    env_parent = cfg.env.env_parent
    env_name = cfg.env.env_name
    task_interface = get_task_interface(task)
    interface_type = task_interface.interface_type
    score_display_name = task_interface.score_display_name

    logging.info(f"Using LLM: {model}")
    logging.info(f"Task: {task}{suffix} (parent={env_parent}, env_name={env_name}, interface={interface_type})")
    logging.info(f"Task description: {task_description}")

    task_file = f"{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py"
    obs_context_file = Path(EUREKA_ROOT_DIR) / task_interface.obs_context_relpath
    if obs_context_file.exists():
        task_obs_code_string = file_to_string(str(obs_context_file))
    else:
        task_obs_code_string = file_to_string(task_file)

    prompt_dir = Path(EUREKA_ROOT_DIR) / "utils" / "prompts"
    bundle = PROMPT_BUNDLE_BY_INTERFACE[interface_type]
    initial_system = file_to_string(str(prompt_dir / bundle["initial_system"]))
    reward_signature = file_to_string(str(prompt_dir / bundle["reward_signature"]))
    policy_feedback = file_to_string(str(prompt_dir / bundle["policy_feedback"]))
    code_output_tip = file_to_string(str(prompt_dir / "code_output_tip.txt"))
    code_feedback = file_to_string(str(prompt_dir / "code_feedback.txt"))
    initial_user = file_to_string(str(prompt_dir / "initial_user.txt"))
    execution_error_feedback = file_to_string(str(prompt_dir / "execution_error_feedback.txt"))

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(
        task_obs_code_string=task_obs_code_string,
        task_description=task_description,
    )
    messages = [
        {"role": "system", "content": initial_system},
        {"role": "user", "content": initial_user},
    ]

    DUMMY_FAILURE = -10000.0
    max_task_scores = []
    max_task_scores_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_task_score_overall = DUMMY_FAILURE
    max_task_score_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None

    for iter in range(cfg.iteration):
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = cfg.sample if "gpt-3.5" in model else 4

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        while True:
            if total_samples >= cfg.sample:
                break
            for attempt in range(1000):
                try:
                    response_cur = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=cfg.temperature,
                        n=chunk_size,
                    )
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        logging.info(f"Current Chunk Size {chunk_size}")
                    logging.info(f"Attempt {attempt + 1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                raise SystemExit(1)

            responses.extend(response_cur["choices"])
            prompt_tokens = response_cur["usage"]["prompt_tokens"]
            total_completion_token += response_cur["usage"]["completion_tokens"]
            total_token += response_cur["usage"]["total_tokens"]

        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0]["message"]["content"] + "\n")

        logging.info(
            f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}"
        )

        code_runs = []
        rl_runs = []
        for response_id in range(cfg.sample):
            response_cur = responses[response_id]["message"]["content"]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            patterns = [
                r"```python(.*?)```",
                r"```(.*?)```",
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            code_string = None
            for pattern in patterns:
                m = re.search(pattern, response_cur, re.DOTALL)
                if m is not None:
                    code_string = m.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
                    break

            try:
                get_function_signature(code_string)
            except Exception as e:
                logging.info(f"Iteration {iter}: Code Run {response_id} cannot parse function signature! err={e!r}")
                try:
                    with open(f"env_iter{iter}_response{response_id}_raw.txt", "w") as fdbg:
                        fdbg.write(response_cur)
                    with open(f"env_iter{iter}_response{response_id}_extracted.py", "w") as fdbg:
                        fdbg.write(code_string)
                except Exception:
                    pass
                code_runs.append("")
                rl_runs.append(None)
                continue

            code_runs.append(code_string)

            gpt_reward_path = str(Path(workspace_dir) / f"gpt_reward_iter{iter}_response{response_id}.py")
            with open(gpt_reward_path, "w") as f:
                f.write("import math\n")
                f.write("import numpy as np\n")
                f.write("from typing import Tuple, Dict\n\n")
                f.write(code_string + "\n")

            with open(f"env_iter{iter}_response{response_id}_rewardonly.py", "w") as f:
                f.write(code_string + "\n")
            shutil.copy(gpt_reward_path, f"env_iter{iter}_response{response_id}.py")

            set_freest_gpu()
            tb_logdir = str(Path(workspace_dir) / f"tb_iter{iter}_response{response_id}")
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            command = [
                "python",
                "-u",
                f"{RL_ROOT_DIR}/train_mujoco.py",
                f"--task={task}{suffix}",
                f"--max_iterations={cfg.max_iterations}",
                f"--seed={iter * cfg.sample + response_id}",
                f"--gpt_reward_module={gpt_reward_path}",
                f"--logdir={tb_logdir}",
                f"--n_envs={cfg.n_envs}",
            ]
            if cfg.use_wandb:
                command.extend(
                    [
                        "--use_wandb",
                        f"--wandb_project={cfg.wandb_project}",
                        f"--wandb_entity={cfg.wandb_username}",
                        f"--wandb_group={workspace_dir.name}",
                        f"--wandb_run_name=iter{iter}_response{response_id}",
                        f"--wandb_job_type=eureka_rl",
                        f"--wandb_tags=eureka,{env_name},iter{iter}",
                    ]
                )
            with open(rl_filepath, "w") as f:
                process = subprocess.Popen(command, stdout=f, stderr=f)
            block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
            rl_runs.append(process)

        contents = []
        task_scores = []
        reward_correlations = []
        code_paths = []
        executed_flags = []

        exec_success = False
        for response_id, rl_run in enumerate(rl_runs):
            code_paths.append(f"env_iter{iter}_response{response_id}.py")
            if rl_run is None:
                content = execution_error_feedback.format(
                    traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!"
                )
                content += code_output_tip
                contents.append(content)
                task_scores.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                executed_flags.append(False)
                continue

            rl_run.communicate()
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            try:
                with open(rl_filepath, "r") as f:
                    stdout_str = f.read()
            except Exception:
                content = execution_error_feedback.format(
                    traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!"
                )
                content += code_output_tip
                contents.append(content)
                task_scores.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                executed_flags.append(False)
                continue

            content = ""
            traceback_msg = filter_traceback(stdout_str)
            if traceback_msg != "":
                task_scores.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                executed_flags.append(False)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)
                content += code_output_tip
                contents.append(content)
                continue

            exec_success = True
            lines = stdout_str.split("\n")
            tensorboard_logdir = None
            for line in lines:
                if line.startswith("Tensorboard Directory:"):
                    tensorboard_logdir = line.split(":", 1)[-1].strip()
                    break
            if tensorboard_logdir is None:
                content = execution_error_feedback.format(traceback_msg="Tensorboard log directory missing from RL stdout.")
                content += code_output_tip
                contents.append(content)
                task_scores.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                executed_flags.append(False)
                continue

            tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
            if "consecutive_successes" not in tensorboard_logs and "gt_reward" not in tensorboard_logs:
                content = execution_error_feedback.format(traceback_msg="Neither " + '"consecutive_successes"' + " nor " + '"gt_reward"' + " was present in Tensorboard logs. Please check whether the RL environment logged a ground-truth task metric.")
                content += code_output_tip
                contents.append(content)
                task_scores.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                executed_flags.append(False)
                continue

            executed_flags.append(True)
            primary_metric_key = "consecutive_successes" if "consecutive_successes" in tensorboard_logs else "gt_reward"
            max_iterations_log = np.array(tensorboard_logs[primary_metric_key]).shape[0]
            epoch_freq = max(int(max_iterations_log // 10), 1)
            content += policy_feedback.format(epoch_freq=epoch_freq)

            reward_correlation = DUMMY_FAILURE
            if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                gt_reward = np.array(tensorboard_logs["gt_reward"])
                gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                if len(gt_reward) > 1 and len(gpt_reward) > 1:
                    reward_correlation = float(np.corrcoef(gt_reward[: len(gpt_reward)], gpt_reward[: len(gt_reward)])[0, 1])
            reward_correlations.append(reward_correlation)

            score_logged = False
            for metric, values in tensorboard_logs.items():
                if "/" in metric:
                    continue
                metric_cur = [f"{x:.2f}" for x in values[::epoch_freq]]
                metric_cur_max = max(values)
                metric_cur_mean = sum(values) / len(values)
                metric_cur_min = min(values)

                if metric == "consecutive_successes":
                    task_scores.append(metric_cur_max)
                    score_logged = True
                    content += f"task_score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                    continue
                if metric == "gt_reward":
                    if "consecutive_successes" not in tensorboard_logs:
                        task_scores.append(metric_cur_max)
                        score_logged = True
                        content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                    continue
                if metric == "gpt_reward":
                    continue
                content += f"{metric}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"

            if not score_logged:
                task_scores.append(DUMMY_FAILURE)
            content += code_feedback
            content += code_output_tip
            contents.append(content)

        if not exec_success and cfg.sample != 1:
            execute_rates.append(0.0)
            max_task_scores.append(DUMMY_FAILURE)
            max_task_scores_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue

        while len(contents) < cfg.sample:
            contents.append(execution_error_feedback.format(
                traceback_msg="Code Run did not produce usable feedback. Please re-write an entirely new reward function!"
            ) + code_output_tip)
        while len(task_scores) < cfg.sample:
            task_scores.append(DUMMY_FAILURE)
            reward_correlations.append(DUMMY_FAILURE)
            executed_flags.append(False)

        best_sample_idx = int(np.argmax(np.array(task_scores)))
        best_content = contents[best_sample_idx]
        max_task_score = task_scores[best_sample_idx]
        max_task_score_reward_correlation = reward_correlations[best_sample_idx]
        execute_rate = float(np.mean(np.array(executed_flags, dtype=np.float64)))

        if max_task_score > max_task_score_overall:
            max_task_score_overall = max_task_score
            max_task_score_reward_correlation_overall = max_task_score_reward_correlation
            max_reward_code_path = code_paths[best_sample_idx]

        execute_rates.append(execute_rate)
        max_task_scores.append(max_task_score)
        max_task_scores_reward_correlation.append(max_task_score_reward_correlation)
        best_code_paths.append(code_paths[best_sample_idx])

        logging.info(
            f"Iteration {iter}: Max Task Score: {max_task_score}, Execute Rate: {execute_rate}, Reward Correlation: {max_task_score_reward_correlation}"
        )
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(f"Iteration {iter}: GPT Output Content:\n" + responses[best_sample_idx]["message"]["content"] + "\n")
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f"{cfg.env.task}")
        x_axis = np.arange(len(max_task_scores))
        axs[0].plot(x_axis, np.array(max_task_scores))
        axs[0].set_title("Max Task Score")
        axs[0].set_xlabel("Iteration")
        axs[1].plot(x_axis, np.array(execute_rates))
        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")
        fig.tight_layout(pad=3.0)
        plt.savefig("summary.png")
        plt.close(fig)
        np.savez(
            "summary.npz",
            max_task_scores=max_task_scores,
            execute_rates=execute_rates,
            best_code_paths=best_code_paths,
            max_task_scores_reward_correlation=max_task_scores_reward_correlation,
        )

        if len(messages) == 2:
            messages += [{"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}]
            messages += [{"role": "user", "content": best_content}]
        else:
            assert len(messages) == 4
            messages[-2] = {"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}
            messages[-1] = {"role": "user", "content": best_content}

        with open("messages.json", "w") as f:
            json.dump(messages, f, indent=4)

    if max_reward_code_path is None:
        logging.info("All iterations of code generation failed, aborting...")
        raise SystemExit(1)

    logging.info(
        f"Task: {task}, Max Training Task Score {max_task_score_overall}, Correlation {max_task_score_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}"
    )
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")

    eval_runs = []
    for i in range(cfg.num_eval):
        set_freest_gpu()
        tb_logdir = str(Path(workspace_dir) / f"tb_eval{i}")
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, "w") as f:
            process = subprocess.Popen(
                [
                    "python",
                    "-u",
                    f"{RL_ROOT_DIR}/train_mujoco.py",
                    f"--task={task}{suffix}",
                    f"--max_iterations={cfg.max_iterations}",
                    f"--seed={i}",
                    f"--gpt_reward_module={max_reward_code_path}",
                    f"--logdir={tb_logdir}",
                    f"--n_envs={cfg.n_envs}",
                ],
                stdout=f,
                stderr=f,
            )
        block_until_training(rl_filepath)
        eval_runs.append(process)

    reward_code_final_scores = []
    reward_code_correlations_final = []
    reward_code_env_returns_final = []
    reward_code_env_episode_returns_final = []
    for i, rl_run in enumerate(eval_runs):
        rl_run.communicate()
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, "r") as f:
            stdout_str = f.read()
        lines = stdout_str.split("\n")
        tb_logdir = None
        for line in lines:
            if line.startswith("Tensorboard Directory:"):
                tb_logdir = line.split(":", 1)[-1].strip()
                break
        if tb_logdir is None:
            continue

        tensorboard_logs = load_tensorboard_logs(tb_logdir)
        if "consecutive_successes" in tensorboard_logs:
            reward_code_final_scores.append(max(tensorboard_logs["consecutive_successes"]))
        elif "gt_reward" in tensorboard_logs:
            reward_code_final_scores.append(max(tensorboard_logs["gt_reward"]))
        if "env_reward" in tensorboard_logs:
            reward_code_env_returns_final.append(max(tensorboard_logs["env_reward"]))
        if "env_episode_return" in tensorboard_logs:
            reward_code_env_episode_returns_final.append(tensorboard_logs["env_episode_return"][-1])
        if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
            gt = np.array(tensorboard_logs["gt_reward"])
            gp = np.array(tensorboard_logs["gpt_reward"])
            if len(gt) > 1 and len(gp) > 1:
                reward_code_correlations_final.append(float(np.corrcoef(gt[: len(gp)], gp[: len(gt)])[0, 1]))

    logging.info(
        f"Final Task Score Mean: {np.mean(reward_code_final_scores) if reward_code_final_scores else float('nan')}, "
        f"Std: {np.std(reward_code_final_scores) if reward_code_final_scores else float('nan')}, Raw: {reward_code_final_scores}"
    )
    logging.info(
        f"Final Env Reward Mean: {np.mean(reward_code_env_returns_final) if reward_code_env_returns_final else float('nan')}, "
        f"Std: {np.std(reward_code_env_returns_final) if reward_code_env_returns_final else float('nan')}, Raw: {reward_code_env_returns_final}"
    )
    logging.info(
        f"Final Env Episode Return Mean: {np.mean(reward_code_env_episode_returns_final) if reward_code_env_episode_returns_final else float('nan')}, "
        f"Std: {np.std(reward_code_env_episode_returns_final) if reward_code_env_episode_returns_final else float('nan')}, Raw: {reward_code_env_episode_returns_final}"
    )
    logging.info(
        f"Final Correlation Mean: {np.mean(reward_code_correlations_final) if reward_code_correlations_final else float('nan')}, "
        f"Std: {np.std(reward_code_correlations_final) if reward_code_correlations_final else float('nan')}, Raw: {reward_code_correlations_final}"
    )
    np.savez(
        "final_eval.npz",
        reward_code_final_scores=reward_code_final_scores,
        reward_code_env_returns_final=reward_code_env_returns_final,
        reward_code_env_episode_returns_final=reward_code_env_episode_returns_final,
        reward_code_correlations_final=reward_code_correlations_final,
    )


if __name__ == "__main__":
    main()
