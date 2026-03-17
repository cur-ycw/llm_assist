"""
SC2SCAREWrapper — SMAC v1 wrapper that injects LLM-generated additional reward
for SCARE encoder pre-training.

Wraps smac.env.StarCraft2Env, intercepts step() to compute and add the
additional reward from AdditionalReward.llm{k}().
"""

from smac.env import StarCraft2Env
from envs.multiagentenv import MultiAgentEnv
from envs.sc2_scare_reward import AdditionalReward


class SC2SCAREWrapper(MultiAgentEnv):
    """SMAC v1 wrapper with pluggable LLM-generated reward for SCARE pre-training."""

    def __init__(self, **kwargs):
        # Pop SCARE-specific arg before passing to StarCraft2Env
        additional_reward_id = kwargs.pop("additional_reward_id", 0)
        self.env = StarCraft2Env(**kwargs)
        self.additional_reward = AdditionalReward(additional_reward_id)

    def reset(self):
        self.additional_reward.reset()
        return self.env.reset()

    def step(self, actions):
        reward, terminated, info = self.env.step(actions)
        # Compute additional reward from LLM-generated function
        self.additional_reward.update(self.env)
        additional = self.additional_reward.calculate()
        info["reward_original"] = reward
        info["reward_additional"] = additional
        reward += additional
        return reward, terminated, info

    def close(self):
        self.env.close()

    def save_replay(self):
        self.env.save_replay()

    def get_obs(self):
        return self.env.get_obs()

    def get_state(self):
        return self.env.get_state()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state_size(self):
        return self.env.get_state_size()

    def get_total_actions(self):
        return self.env.get_total_actions()

    def get_obs_agent(self, agent_id):
        return self.env.get_obs_agent(agent_id)

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.env.n_agents,
            "n_enemies": self.env.n_enemies,
            "episode_limit": self.env.episode_limit,
        }
        return env_info

    def __getattr__(self, name):
        if hasattr(self.env, name):
            return getattr(self.env, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
