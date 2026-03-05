prompt_basic_env = 'You are an expert in cooperative multi-agent reinforcement learning (MARL) and code generation. We are going to train a team of two players in the Predator-Prey (PP) game. The game is a 2D world with two predators and five prey (two stags S1 S2 and three rabbits R1 R2 R3). Each player controls a predator. They need to choose the prey to catch (like S1 or R2+R3), then chase the chosen prey to catch it / them. Stags require two predators to catch at the same time. If only one predator is near them, both players will be punished. Rabbits only require one predator to catch them. When players successfully catch a stag, they get reward 1. When players successfully catch a rabbit, they get reward 0.5.\n'

prompt_multi_modality = "\nHuman player teams may have specific cooperation preferences to play the game, like catching stag S1. They have their own `additional_reward` shown in the code. A new player outside a team needs to learn and adapt these preferences to cooperate well after joining the team.\n"

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
    self.predators_position : {"1": np.ndarray[(2,)], "2": np.ndarray[(2,)]} # Initialization: both np.random.uniform(-0.1, +0.1, 2)
    self.prey_position : {"S1": np.ndarray[(2,)], "S2": np.ndarray[(2,)], "R1": np.ndarray[(2,)], "R2": np.ndarray[(2,)], "R3": np.ndarray[(2,)]} # Initialization: "S1": [1., 0.], "S2": [-1., 0.], "R1": [0.8, 0.6], "R2": [-0.8, 0.6], "R3": [0., -1.]
    self.caught_prey_set = set() # record the prey caught by the team, like {"S1"} or {"R2", "R3"}, and an empty set means no prey has been caught yet.
    def entity_distance(self, entity1 : str, entity2 : str) -> float:
        # return the distance between the input entities, like "1" and "2", "1" and "S1", "R1" and "R2", etc.
    def get_prey_level(self, prey : str) -> int:
        # return 2 for "S1" and "S2", return 1 for "R1" and "R2" and "R3"
    def get_num_predator_nearby(self, prey : str) -> int:
        # return the number of predators near / catching the prey (distance <= 0.25), can be 0 or 1 or 2
    def step(self):
        # other codes that change positions
        reward = 0.0
        for prey in self.prey_position.keys():
            prey_level = self.get_prey_level(prey)
            num_predator_nearby = self.get_num_predator_nearby(prey)
            if num_predator_nearby == 0:
                continue
            elif 0 < num_predator_nearby < prey_level:
                reward -= 0.01
            if num_predator_nearby >= prey_level:
                reward += prey_level / 2
                self.caught_prey_set.add(prey)
        reward += self.additional_reward()
        # other codes
```
'''
