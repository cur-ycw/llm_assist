prompt_basic_env = 'You are an expert in cooperative multi-agent reinforcement learning (MARL) and code generation. We are going to train a team of two football players (Turing and Johnson) in the Google Research Football (GRF) game. They try to score from the edge of the box, Johnson is on the side with the ball, Turing is at the center and facing the goalkeeper (Meitner). Our team gets reward 1 when scoring a goal. An episode ends when our team scores a goal, or Meitner owns the ball, or the ball is out of bounds.\n'

prompt_multi_modality = "\nHuman player teams may have specific cooperation preferences to play the game, like scoring goals by a specific player. They have their own `additional_reward` shown in the code. A new player outside a team needs to learn and adapt these preferences to cooperate well after joining the team.\n"

prompt_behavior_1 =  "\nBased on the information above, think step by step to come up with "
prompt_behavior_first = "a "
prompt_behavior = "another "
prompt_behavior_2 = "possible cooperation preference. The preference should be deterministic and concrete. It should be as simple as possible. Avoid conditional terms like if, unless, when, etc. It should be easily implemented in python codes using the provided code snippet. It should not conflict with the original task objective.\n\n" + \
    "Finally, output the preference in the format: 'Human players may prefer to \{preference\}'"
prompt_behavior_first = prompt_behavior_1 + prompt_behavior_first + prompt_behavior_2
prompt_behavior = prompt_behavior_1 + prompt_behavior + prompt_behavior_2

prompt_write_code = '''
According to this cooperation preference, write an operational and executable reward function that formats as 'def additional_reward(self) -> float' and returns the 'reward : float' only.

1. Please think step by step and tell us what this code means.
2. The code function must align with the cooperation preference. Do not encourage learning of other behaviors.
3. Short and simple code is better.
4. Consider return a large (tens or even hundreds) reward only when a goal is scored (`if self.score`).
'''

prompt_code = '''
Here's a part of the original code:

```python
class Game:
    ## 1. Location information
    # The closer to the opponent's goal, the larger the x-coordinate. The y-coordinate of the left half of the field is < 0, and the y-coordinate of the right half is > zero.
    self.ball_position : np.ndarray[(2,)] # ball's (x, y) coordinate, (0.7, -0.28) at the beginning
    self.Turing_position : np.ndarray[(2,)] # Turing's (x, y) coordinate, (0.7, 0.0) at the beginning
    self.Johnson_position : np.ndarray[(2,)] # Johnson's (x, y) coordinate, (0.7, -0.3) at the beginning
    self.Meitner_position : np.ndarray[(2,)] # Meitner' (x, y) coordinate, (1.0, 0.0) at the beginning
    # Coordinates of the lower left and right corners of the goal are about (1.0, -0.04) and (1.0, 0.04)
    ## 2. Critical game-level information
    self.pass_history : list # List to store the history of passes as tuples, with the first element as the player who made the pass and the second element as the player who received it, for example, [("Johnson", "Turing"), ("Turing", "Johnson")]
    self.score : bool # True if the team scores a goal at this step and False otherwise
    self.score_Turing : bool # True if Turing scores a goal at this step and False otherwise
    self.score_Johnson : bool # True if Johnson scores a goal at this step and False otherwise
    def step(self):
        # other codes that change the above attributes
        reward = 0.0
        if self.score:
            reward += 1
        reward += self.additional_reward()
        # other codes
```
'''
