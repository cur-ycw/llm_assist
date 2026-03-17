"""
AdditionalReward for SCARE pre-training on SMAC v1.

Extracts agent/enemy positions from StarCraft2Env.get_state_dict() and
provides helper methods that LLM-generated reward functions can use.

LLM-generated methods (llm0, llm1, ...) are appended to this class by
the pretrain pipeline. The `additional_reward_id` selects which one to call.
"""

import numpy as np


class AdditionalReward:
    """Computes LLM-generated additional reward for SCARE encoder pre-training."""

    def __init__(self, additional_reward_id=0):
        self.additional_reward_id = additional_reward_id
        self.agents_position = {}
        self.enemies_position = {}
        self.killed_enemy = ""
        self._prev_death_tracker = None

        # Discover available llm{k} methods
        self._reward_fn = None
        fn_name = f"llm{self.additional_reward_id}"
        if hasattr(self, fn_name):
            self._reward_fn = getattr(self, fn_name)

    def reset(self):
        """Reset state at episode start."""
        self.agents_position = {}
        self.enemies_position = {}
        self.killed_enemy = ""
        self._prev_death_tracker = None

    def update(self, env):
        """Extract positions and killed-enemy info from the SMAC v1 env.

        Args:
            env: smac.env.StarCraft2Env instance (the inner unwrapped env)
        """
        state_dict = env.get_state_dict()
        ally_state = state_dict["allies"]    # (n_agents, nf_al)
        enemy_state = state_dict["enemies"]  # (n_enemies, nf_en)

        # Agent labels: "1", "2", "3", ...
        for i in range(ally_state.shape[0]):
            self.agents_position[str(i + 1)] = ally_state[i, 2:4].copy()

        # Enemy labels: "A", "B", "C", ...
        enemy_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i in range(enemy_state.shape[0]):
            self.enemies_position[enemy_labels[i]] = enemy_state[i, 1:3].copy()

        # Track newly killed enemies by diffing death_tracker_enemy
        current_deaths = env.death_tracker_enemy.copy()
        if self._prev_death_tracker is not None:
            new_kills = current_deaths - self._prev_death_tracker
            for i in range(len(new_kills)):
                if new_kills[i] > 0:
                    self.killed_enemy = enemy_labels[i]
        self._prev_death_tracker = current_deaths

    def calculate(self):
        """Call the selected LLM reward function. Returns 0.0 if none available."""
        if self._reward_fn is None:
            # Re-check in case method was appended after __init__
            fn_name = f"llm{self.additional_reward_id}"
            if hasattr(self, fn_name):
                self._reward_fn = getattr(self, fn_name)
            else:
                return 0.0
        return self._reward_fn()

    def agent_enemy_distance(self, agent_idx, enemy_idx):
        """Euclidean distance between an agent and an enemy (normalized coords)."""
        agent_pos = self.agents_position[agent_idx]
        enemy_pos = self.enemies_position[enemy_idx]
        return np.linalg.norm(agent_pos - enemy_pos)
