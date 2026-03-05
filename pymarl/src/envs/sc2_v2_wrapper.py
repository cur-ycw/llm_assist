from smacv2.env.starcraft2.distributions import get_distribution
from smacv2.env.starcraft2.starcraft2 import StarCraft2Env, CannotResetException
from envs.multiagentenv import MultiAgentEnv


class StarCraftCapabilityEnvWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.distribution_config = kwargs["capability_config"]
        self.env_key_to_distribution_map = {}
        self._parse_distribution_config()
        self.env = StarCraft2Env(**kwargs)
        assert (
            self.distribution_config.keys()
            == kwargs["capability_config"].keys()
        ), "Must give distribution config and capability config the same keys"

    def _parse_distribution_config(self):
        for env_key, config in self.distribution_config.items():
            if env_key == "n_units" or env_key == "n_enemies":
                continue
            config["env_key"] = env_key
            config["n_units"] = self.distribution_config["n_units"]
            config["n_enemies"] = self.distribution_config["n_enemies"]
            distribution = get_distribution(config["dist_type"])(config)
            self.env_key_to_distribution_map[env_key] = distribution

    def reset(self):
        try:
            reset_config = {}
            for distribution in self.env_key_to_distribution_map.values():
                reset_config = {**reset_config, **distribution.generate()}
            return self.env.reset(reset_config)
        except CannotResetException:
            return self.reset()

    def __getattr__(self, name):
        if hasattr(self.env, name):
            return getattr(self.env, name)
        raise AttributeError

    def get_obs(self):
        return self.env.get_obs()

    def get_state(self):
        return self.env.get_state()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_env_info(self):
        return self.env.get_env_info()

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

    def step(self, actions):
        return self.env.step(actions)

    def save_replay(self):
        self.env.save_replay()

    def close(self):
        self.env.close()


class StarCraft2Env2Wrapper(StarCraftCapabilityEnvWrapper):
    """sc2_v2 env wrapper for pymarl; extends capability wrapper with get_env_info for pymarl."""

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.env.n_agents,
            "n_enemies": getattr(self.env, "n_enemies", self.env.n_agents),
            "episode_limit": self.env.episode_limit,
        }
        return env_info
