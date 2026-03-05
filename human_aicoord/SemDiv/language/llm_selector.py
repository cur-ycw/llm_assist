from call_llm import LLM
import re

class LLMSelector:
    def __init__(self, env):
        self.llm = LLM(mode='openai')
        self.big_model = True
        self.env = env
        
    def select_head(self, human_description, head2description):
        
        if 'lbf' in self.env:
            prompt = "You are an expert in the Level-Based Foraging (LBF) game. "
            prompt += "We are going to build a team of two players 1 and 2 in the LBF game. "
            prompt += "They need to select and collect a food together. There are four foods (A, B, C, D) in the field. "
            prompt += "foods_position : A = [0, 0], B = [0, 7], C = [7, 0], D = [7, 7] "
            prompt += "\nPlayer 1 was trained under the same situation, but with different teammates other than Player 2 to achieve the following cooperation preferences, and learned corresponding policies:\n"
            for head, description in head2description.items():
                description_ = description.replace('**', '').replace('"', '').replace("'", "")
                prompt += f'{head}: {description_}\n'
            prompt += f'Now, Player 2 says that: "{human_description}" '
            prompt += 'Based on the information above, please carefully analyze the game, the predators, the prey, etc. Think step by step to select the policy (1~6) for Player 1 that can best coordinate with Player 2 and satisfy his preferences. Output your answer in the format "[n]". For example, if your answer is policy 3, output "[3]".'
        elif 'pp' in self.env:
            prompt = "You are an expert in the predator-prey (PP) game. "
            prompt += "We are going to build a team of two players 1 and 2 controlling two predators in the PP game. "
            prompt += "They need to chase and catch the prey. There are five prey including two stags (S1, S2) and three rabbits (R1, R2, R3). "
            prompt += "Stags require two predators to catch at the same time. Rabbits only require one predator to catch them. "
            prompt += "\nPlayer 1 was trained under the same situation, but with different teammates other than Player 2 to achieve the following cooperation preferences, and learned corresponding policies:\n"
            for head, description in head2description.items():
                description_ = description.replace('**', '').replace('"', '').replace("'", "")
                prompt += f'{head}: {description_}\n'
            prompt += f'Now, Player 2 says that: "{human_description}" '
            prompt += 'Based on the information above, please carefully analyze the game, the predators, the prey, etc. Think step by step to select the policy (1~6) for Player 1 that can best coordinate with Player 2 and satisfy his preferences. Output your answer in the format "[n]". For example, if your answer is policy 3, output "[3]".'
        elif 'sc2' in self.env:
            prompt = "You are an expert in the Starcraft Multi-Agent Challenge (SMAC) game. "
            prompt += "We are going to build a team of two players 1 and 2 controlling two marines in the SMAC game. "
            prompt += 'They need to beat four enemy marines ("A", "B", "C", and "D"). The two marine agents are spawned at the center of the field, and four enemies are scattered in four different corners. '
            prompt += 'Initial positions: agents near [0., 0.], "A" lower left, "B" upper left, "C" upper right, "D" bottom right. '
            prompt += 'They need to choose a same enemy, move towards it, and fire at it together to kill it. '
            prompt += "\nPlayer 1 was trained under the same situation, but with different teammates other than Player 2 to achieve the following cooperation preferences, and learned corresponding policies:\n"
            for head, description in head2description.items():
                description_ = description.replace('**', '').replace('"', '').replace("'", "")
                prompt += f'{head}: {description_}\n'
            prompt += f'Now, Player 2 says that: "{human_description}" '
            prompt += 'Based on the information above, please carefully analyze the game, the predators, the prey, etc. Think step by step to select the policy (1~6) for Player 1 that can best coordinate with Player 2 and satisfy his preferences. Output your answer in the format "[n]". For example, if your answer is policy 3, output "[3]".'
        elif self.env == 'football1':
            prompt = "You are an expert in football. "
            prompt += "We are going to build a team of two football players (Turing and Johnson, no other teammates). "
            prompt += "They need to score from the edge of the box. When the game starts, Johnson is on the left side controlling the ball, Turing is at the center and facing the goalkeeper. "
            prompt += "\nJohnson was trained under the same situation, but with different teammates other than Turing to achieve the following cooperation preferences, and learned corresponding policies:\n"
            for head, description in head2description.items():
                description_ = description.replace('**', '').replace('"', '').replace("'", "")
                prompt += f'{head}: {description_}\n'
            prompt += f'Now, Turing says that: "{human_description}" '
            prompt += 'Based on the information above, please carefully analyze the game, the ball, the policies, etc. Think step by step to select the policy (1~6) for Johnson that can best coordinate with Turing and satisfy his preferences. Output your answer in the format "[n]". For example, if your answer is policy 3, output "[3]".'
        elif self.env == 'football2':
            prompt = "You are an expert in football. "
            prompt += "We are going to build a team of two football players (Turing and Johnson, no other teammates). "
            prompt += "They need to score from the edge of the box. When the game starts, Johnson is on the left side controlling the ball, Turing is at the center and facing the goalkeeper. "
            prompt += "\nTuring was trained under the same situation, but with different teammates other than Johnson to achieve the following cooperation preferences, and learned corresponding policies:\n"
            for head, description in head2description.items():
                description_ = description.replace('**', '').replace('"', '').replace("'", "")
                prompt += f'{head}: {description_}\n'
            prompt += f'Now, Johnson says that: "{human_description}" '
            prompt += 'Based on the information above, please carefully analyze the game, the ball, the policies, etc. Think step by step to select the policy (1~6) for Turing that can best coordinate with Johnson and satisfy his preferences. Output your answer in the format "[n]". For example, if your answer is policy 3, output "[3]".'
        else:
            assert 0
        
        print(prompt)
        print('-' * 50)
        while True:
            output = self.llm.call_llm(prompt, big_model=self.big_model)
            pattern = r'\[(\d+)\]'
            matches = re.findall(pattern, output)
            if len(set(matches)) == 1:
                selected_head = int(matches[0])
                if selected_head in [1, 2, 3, 4, 5, 6]:
                    print(output)
                    return selected_head