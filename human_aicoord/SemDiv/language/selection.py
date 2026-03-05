import json
import numpy as np
import os
from llm_selector import LLMSelector

env = ['lbf', 'pp', 'sc2', 'football1', 'football2'][0]

env2humans_different_descriptions = {
    'lbf': {
        1: ["I prefer to collect food A."],
        2: ["I prefer to collect food B."],
        3: ["I prefer to collect food C."],
        4: ["I prefer to collect food D."],
        5: ["I prefer to collect the food closest to our average position."],
    },
    'pp': {
        1: ["I prefer to catch stag S1."],
        2: ["I prefer to catch stag S2."],
        3: ["I prefer to catch rabbit R1 and R2."],
        4: ["I prefer to catch rabbit R1 and R3."],
        5: ["I prefer to catch rabbit R2 and R3."],
    },
    'sc2': {
        1: ["I prefer to kill enemy A."],
        2: ["I prefer to kill enemy B."],
        3: ["I prefer to kill enemy C."],
        4: ["I prefer to kill enemy D."],
        5: ["I prefer to kill the closet enemy."],
    },
    'football1': {
        1: ["I prefer Johnson to score."],
        2: ["I prefer to score myself."],
        3: ["I prefer that we don't pass the ball."],
        4: ["I prefer that we pass once before shooting."],
        5: ["I prefer that we pass twice before shooting."],
    },
    'football2': {
        1: ["I prefer to score myself."],
        2: ["I prefer Turing to score."],
        3: ["I prefer that we don't pass the ball."],
        4: ["I prefer that we pass once before shooting."],
        5: ["I prefer that we pass twice before shooting."],
    }
}

env2seed2lib = {
    'lbf':{
        1: 'language/lib/2025-11-24_07-15-05_semdiv', # fill the lib directory
    },
    'pp':{
        1: 'language/lib/xxx',
    },
    'sc2':{
        1: 'language/lib/xxx',
    },
    'football1':{
        1: 'language/lib/xxx',
    },
    'football2':{
        1: 'language/lib/xxx',
    },
}
seed2lib = env2seed2lib[env]
seed2head2description = {}
for seed, lib in seed2lib.items():
    seed2head2description[seed] = {}
    ego_lib = os.path.join(lib, 'ego.json')
    behavior_lib = os.path.join(lib, 'behavior.json')
    with open(ego_lib, encoding="utf-8") as file:
        ego_info = json.load(file)
    with open(behavior_lib, encoding="utf-8") as file:
        all_behavior_info = json.load(file)
    for head_idx, head_info in ego_info.items():
        seed2head2description[seed][int(head_idx)] = head_info['behavior']
    
seed2head2results = {}
if env == 'lbf':
    d_tmp = {
        "A": "1",
        "B": "2",
        "C": "3",
        "D": "4",
        "E": "5",
    }
    with open("2025-11-24_07-15-05_semdiv/res.json", encoding="utf-8") as file:
        data = json.load(file)
    ego_idx = 0
    for seed in [1, 2, 3]:
        seed2head2results[seed] = {}
        for run, res in data.items():
            player0_, player1_ = run.split('+++')
            player0 = player0_
            player1 = d_tmp[player1_.split('_')[-3]]
            if ego_idx == 0:
                human, head = (player1, player0)
            elif ego_idx == 1:
                human, head = (player0, player1)
            human_id = int(human)
            head_id = int(head.split('---')[1]) + 1
            if head_id not in seed2head2results[seed]:
                seed2head2results[seed][head_id] = {}
            seed2head2results[seed][head_id][human_id] = (round(res['return_original'], 2), round(res['desired_ratio'], 2))
elif env == 'pp':
    with open("xxx/res.json", encoding="utf-8") as file:
        data = json.load(file)
    ego_idx = 0
    for seed in [1, 2, 3]:
        seed2head2results[seed] = {}
        for run, res in data.items():
            player0_, player1_ = run.split('+++')
            # player0 = player0_.split('---')[1]
            player0 = player0_
            player1 = player1_.split('_human')[1][0]
            if ego_idx == 0:
                human, head = (player1, player0)
            elif ego_idx == 1:
                human, head = (player0, player1)
            human_id = int(human)
            head_id = int(head.split('---')[1]) + 1
            if head_id not in seed2head2results[seed]:
                seed2head2results[seed][head_id] = {}
            seed2head2results[seed][head_id][human_id] = (round(res['return_original'], 2), round(res['desired_ratio'], 2))
elif env == 'sc2':
    d_tmp = {
        "A": "1",
        "B": "2",
        "C": "3",
        "D": "4",
        "E": "5",
    }
    with open("xxx/res.json", encoding="utf-8") as file:
        data = json.load(file)
    ego_idx = 0
    for seed in [1, 2]:
        seed2head2results[seed] = {}
        for run, res in data.items():
            player0_, player1_ = run.split('+++')
            # player0 = player0_.split('---')[1]
            player0 = player0_
            player1 = d_tmp[player1_.split('_')[-3]]
            if ego_idx == 0:
                human, head = (player1, player0)
            elif ego_idx == 1:
                human, head = (player0, player1)
            human_id = int(human)
            head_id = int(head.split('---')[1]) + 1
            if head_id not in seed2head2results[seed]:
                seed2head2results[seed][head_id] = {}
            seed2head2results[seed][head_id][human_id] = (round(res['return_original'], 2), round(res['desired_ratio'], 2))
elif 'football' in env:
    if env == 'football1':
        path = "xxx/res.json"
    if env == 'football2':
        path = "xxx/res.json"
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if env == 'football1':
        ego_idx = 1
    if env == 'football2':
        ego_idx = 0
    for seed in range(1, len(env2seed2lib[env]) + 1):
        seed2head2results[seed] = {}
        for run, res in data.items():
            player0_, player1_ = run.split('+++')
            if ego_idx == 0:
                player0_, player1_ = player1_, player0_
            elif ego_idx == 1:
                pass
            player0 = player0_.split('/')[-3].split('_')[-1]
            if '/' in player1_:
                player1 = player1_.split('/')[-3]
            else:
                player1 = player1_
            human, head = (player0, player1)
            human_id = int(human)
            head_id = int(head.split('_')[-1])
            if head_id not in seed2head2results[seed]:
                seed2head2results[seed][head_id] = {}
            seed2head2results[seed][head_id][human_id] = (round(res['eval_score_rate'], 2), round(res['eval_average_episode_rewards_additional'], 2))


llm_selector = LLMSelector(env)
    
seed2human2results = {}
for seed in seed2lib.keys():
    seed2human2results[seed] = {}
    head2description = seed2head2description[seed]
    for human, description_lst in env2humans_different_descriptions[env].items():
        seed2human2results[seed][human] = []
        for description in description_lst:
            selected_head = llm_selector.select_head(description, head2description)
            print('\nResult:', description, '|', head2description[selected_head], '\n')
            res = seed2head2results[seed][selected_head][human]
            seed2human2results[seed][human].append(res)
print()

n_seed = len(seed2human2results.keys())
n_human = len(env2humans_different_descriptions[env])
arr = np.zeros((n_seed, n_human))
arr_ = np.zeros((n_seed, n_human))
for i, seed in enumerate(seed2human2results.keys()):
    print(seed, seed2human2results[seed])
    for j, human in enumerate(sorted(seed2human2results[seed].keys())):
        arr[i][j] = seed2human2results[seed][human][0][0]
        arr_[i][j] = seed2human2results[seed][human][0][1]
print(arr)