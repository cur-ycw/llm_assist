prompt_basic_env = 'You are an expert in cooperative multi-agent reinforcement learning (MARL) and code generation. We are going to train a team of two players in the Level-Based Foraging (LBF) game. The game is a 2D square grid-world with two agents, and four foods (denoted as food "A", "B", "C", and "D") are scattered in four different corners. Each player controls an agent. They need to choose a same food and move towards it, and be at adjacent grids of it together to collect the food. When agents successfully collect the first food, like food "B", they get reward 1 and the game ends.\n'

prompt_multi_modality = "\nHuman player teams may have specific cooperation preferences to play the game, like collecting different foods. They have their own `additional_reward` shown in the code. A new player outside a team needs to learn and adapt these preferences to cooperate well after joining the team.\n"

prompt_behavior_1 =  "\nBased on the information above, think step by step to come up with "
prompt_behavior_first = "a "
prompt_behavior = "another "
prompt_behavior_2 = "possible cooperation preference. The preference should be deterministic and concrete. It should be as simple as possible. Avoid conditional terms like if, unless, when, etc. Avoid sequential behaviors like 'first X, then Y'. It should be easily implemented in python codes using the provided code snippet. It should not conflict with the original task objective.\n\n" + \
    "Finally, output the preference in the format: 'Human players may prefer to \{preference\}'"
prompt_behavior_first = prompt_behavior_1 + prompt_behavior_first + prompt_behavior_2
prompt_behavior = prompt_behavior_1 + prompt_behavior + prompt_behavior_2

prompt_write_code = '''
According to this cooperation preference, write an operational and executable reward function that formats as 'def additional_reward(self) -> float' and returns the 'reward : float' only.

1. Please think step by step and tell us what this code means;
2. The code function must align with the cooperation preference.
3. It can be a dense reward that guides the team to learn the cooperation preference.
4. Short and simple code is better.
'''

prompt_code = '''
Here's a part of the original code:

```python
class ForagingEnv(Env):
    self.agents_position : {"1": np.ndarray[(2,)], "2": np.ndarray[(2,)]}
    self.foods_position : {"A": np.array([0, 0]), "B": np.array([0, 7]), "C": np.array([7, 0]), "D": np.array([7, 7])}
    self.collected_food : str # record the food ("A" / "B" / "C" / "D" / "") collected by the team, and "" means no food has been collected yet.
    # other attributes and functions
    
    def agent_food_distance(self, agent_idx: str, food_idx: str):
        agent_pos = self.agents_position[agent_idx]
        food_pos = self.foods_position[food_idx]
        distance = np.linalg.norm(agent_pos - food_pos)
        return distance

    def step(self):
        # other codes
        reward = 0
        # process collectings: if agents successfully collect one food, reward = 1
        for food, (food_row, food_col) in self.foods_position.items():
            # 2 agents be at adjacent grids of it together to collect the food
            n_adj_players = self.adjacent_player_number(food_row, food_col)
            if n_adj_players == 2:
                self.collected_food = food
                reward = 1
                break
        # when agents successfully collect a food, they get reward = 1 and the game ends.
        done = (reward == 1) or (self.current_step >= self._max_episode_steps)
        reward += self.additional_reward()
        # return new state, reward, done, and other step info
```
'''
