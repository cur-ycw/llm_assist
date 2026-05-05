from collections import defaultdict
import logging
import os
import numpy as np

def _to_scalar(x):
    """Convert tensor/ndarray to Python float for logging."""
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return float(x.item()) if x.size == 1 else float(np.mean(x))
    return float(x)

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False
        self.use_wandb = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        """使用 TensorBoard 记录指标：tensorboard-logger 写入，无需 TensorFlow。"""
        try:
            from tensorboard_logger import configure, log_value
            configure(directory_name)
            self.tb_logger = log_value
            self.use_tb = True
        except ImportError:
            self.console_logger.warning(
                "tensorboard-logger 未安装，无法记录到 TensorBoard。请安装: pip install tensorboard-logger"
            )

    def close_tb(self):
        """TensorBoard 写入结束（tensorboard-logger 无显式关闭，保留接口兼容）。"""
        pass

    def setup_wandb(self, project, run_name, config=None, entity=None, group=None, tags=None):
        try:
            import wandb
            mode = os.environ.get("WANDB_MODE", "online")
            init_timeout = int(os.environ.get("WANDB_INIT_TIMEOUT", "180"))
            start_method = os.environ.get("WANDB_START_METHOD", "thread")
            wandb.init(
                project=project,
                name=run_name,
                config=config,
                entity=entity,
                group=group,
                tags=tags,
                mode=mode,
                reinit=True,
                settings=wandb.Settings(init_timeout=init_timeout, start_method=start_method),
            )
            self.wandb = wandb
            self.use_wandb = True
        except ImportError:
            self.console_logger.warning(
                "wandb 未安装，无法记录到 Weights & Biases。请安装: pip install wandb"
            )
        except Exception as exc:
            self.console_logger.warning("Weights & Biases 初始化失败: {}".format(exc))

    def close_wandb(self):
        if self.use_wandb and getattr(self, "wandb", None) is not None:
            try:
                self.wandb.finish()
            except Exception:
                pass

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb and getattr(self, "tb_logger", None) is not None:
            try:
                v = _to_scalar(value)
                step = int(_to_scalar(t))
                self.tb_logger(key, v, step)
            except Exception:
                pass

        if self.use_wandb and getattr(self, "wandb", None) is not None:
            try:
                self.wandb.log({key: _to_scalar(value), "trainer_step": int(_to_scalar(t))}, step=int(_to_scalar(t)))
            except Exception:
                pass

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def print_recent_stats(self):
        t_env, episode = self.stats["episode"][-1]
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(_to_scalar(t_env), _to_scalar(episode))
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            values = [_to_scalar(x[1]) for x in self.stats[k][-window:]]
            item = "{:.4f}".format(np.mean(values))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger
