prompt_basic_env = 'You are an expert in cooperative multi-agent reinforcement learning (MARL) and code generation. We are going to train a team of two players in the Starcraft Multi-Agent Challenge (SMAC) game, which involves unit micromanagement tasks. In this game, ally units need to beat enemy units controlled by the built-in AI. Specifically, each player controlls a marine agent ("1" and "2") to beat four enemy marines ("A", "B", "C", and "D"). The two marine agents are spawned at the center of the field, and four enemies are scattered in four different corners. Agents need to choose a same enemy, move towards it, and fire at it together to kill it. When agents successfully kill the first enemy, like enemy "B", they get a reward about 10 and the game ends. If both agents are killed, they lose.\n'

prompt_multi_modality = "\nHuman player teams may have specific cooperation preferences to play the game, like attacking different enemies. They have their own `additional_reward` shown in the code. A new player outside a team needs to learn and adapt these preferences to cooperate well after joining the team.\n"

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
class Game:
    self.agents_position : {"1": np.ndarray[(2,)], "2": np.ndarray[(2,)]}
    self.enemies_position : {"A": np.ndarray[(2,)], "B": np.ndarray[(2,)], "C": np.ndarray[(2,)], "D": np.ndarray[(2,)]}
    # these 2D positions are calculated as [(x - self.center_x) / self.max_distance_x, (y - self.center_y) / self.max_distance_y]
    # initial positions: agents near [0., 0.], "A" lower left, "B" upper left, "C" upper right, "D" bottom right 
    # for agents and enemies that are killed, their postions will be set to [0., 0.]
    self.killed_enemy : str # record the enemy ("A" / "B" / "C" / "D" / "") killed by the team, and "" means no enemy has been killed yet.
    # other attributes and functions
    
    def agent_enemy_distance(self, agent_idx: str, enemy_idx: str):
        agent_pos = self.agents_position[agent_idx]
        enemy_pos = self.enemies_position[enemy_idx]
        distance = np.linalg.norm(agent_pos - enemy_pos)
        return distance

    def step(self):
        reward = 0.0
        # other codes that change the battle state the above attributes, and calculate the original reward
        reward += self.additional_reward()
        # other codes
```
'''
