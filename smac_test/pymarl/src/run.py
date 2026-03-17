import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # TensorBoard 记录：日志目录 results/tensorboard，由 tensorboard-logger 写入（非 TensorFlow）
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_root = os.path.join(dirname(dirname(abspath(__file__))), "results", "tensorboard")
        os.makedirs(tb_root, exist_ok=True)
        tb_exp_direc = os.path.join(tb_root, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(0)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    # --- SCARE: load pretrained encoders + alpha_prior ---
    if args.learner == "scare_learner" and getattr(args, "scare_lib_dir", ""):
        import json
        from scare_pretrain import load_pretrained_encoders
        # Load pretrained encoder weights into SCARERNNAgent
        load_pretrained_encoders(mac.agent, args.scare_lib_dir)
        logger.console_logger.info(
            "Loaded pretrained encoders from {}".format(args.scare_lib_dir)
        )
        # Load alpha_prior and store on args for learner to pick up
        prior_path = os.path.join(args.scare_lib_dir, 'alpha_prior.json')
        if os.path.exists(prior_path):
            with open(prior_path, 'r') as f:
                args.alpha_prior = json.load(f)
            # Rebuild learner's alpha_prior tensor
            learner._build_alpha_prior(args.alpha_prior)
            logger.console_logger.info(
                "Loaded alpha_prior from {}".format(prior_path)
            )

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time
    _logged_train_device = False

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # Timing accumulators for profiling collect vs train
    _t_collect_total = 0.0
    _t_train_total = 0.0
    _n_loops = 0

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        _t0 = time.time()
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)
        _t_collect_total += time.time() - _t0

        if buffer.can_sample(args.batch_size):
            # training_iters: how many gradient steps per data collection.
            # Default 1 (standard). Increase when using parallel runner with
            # many envs (e.g. 16 for batch_size_run=64) to improve sample efficiency.
            n_train = getattr(args, "training_iters", 1)
            _t1 = time.time()
            for _ in range(n_train):
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                if not _logged_train_device and args.use_cuda:
                    _dev = next(learner.mac.parameters()).device
                    _batch_dev = episode_sample.data.transition_data["state"].device
                    logger.console_logger.info("Train device check: model on {}, batch on {} (expect cuda for both)".format(_dev, _batch_dev))
                    _logged_train_device = True

                learner.train(episode_sample, runner.t_env, episode)
            _t_train_total += time.time() - _t1

        _n_loops += 1

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            # Print collect vs train time breakdown
            _t_total = _t_collect_total + _t_train_total
            if _t_total > 0:
                logger.console_logger.info(
                    "Time breakdown | collect: {:.1f}s ({:.0f}%) | train(x{}): {:.1f}s ({:.0f}%) | loops: {}".format(
                        _t_collect_total, 100 * _t_collect_total / _t_total,
                        getattr(args, "training_iters", 1),
                        _t_train_total, 100 * _t_train_total / _t_total,
                        _n_loops))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    if getattr(logger, "use_tb", False):
        logger.close_tb()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")
    elif config["use_cuda"]:
        _log.info("Using device: cuda (GPU). Training will use GPU after first {} episodes are collected.".format(config["batch_size"]))

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
