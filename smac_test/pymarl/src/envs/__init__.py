from functools import partial
import sys
import os

# Base MultiAgentEnv from local (used by sc2_v2); smac.env.MultiAgentEnv not imported
# here to avoid loading smac maps when only using sc2_v2 (would cause pysc2 DuplicateMapError).
from envs.multiagentenv import MultiAgentEnv

def env_fn(env, **kwargs):
    return env(**kwargs)

REGISTRY = {}

# sc2 (SMAC v1): lazy import so smac maps are not registered when only running sc2_v2
def _make_sc2(**kwargs):
    from smac.env import StarCraft2Env
    return StarCraft2Env(**kwargs)

REGISTRY["sc2"] = _make_sc2

# SMAC v2 (sc2_v2): imports smacv2 (and its maps) only when this env is used
from envs.sc2_v2_wrapper import StarCraft2Env2Wrapper
REGISTRY["sc2_v2"] = partial(env_fn, env=StarCraft2Env2Wrapper)

# SMAC v1 + SCARE additional reward wrapper
from envs.sc2_scare_wrapper import SC2SCAREWrapper
REGISTRY["sc2_scare"] = partial(env_fn, env=SC2SCAREWrapper)

if sys.platform == "linux":
    _src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    _sc2_3rdparty = os.path.join(_src_dir, "3rdparty", "StarCraftII")
    if os.path.isdir(_sc2_3rdparty):
        os.environ.setdefault("SC2PATH", _sc2_3rdparty)
    else:
        os.environ.setdefault("SC2PATH",
                              os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
