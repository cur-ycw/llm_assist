"""
SCARE Encoder Pre-training Pipeline

Adapts SemDiv's LLM-driven semantic diversity approach to pre-train K frozen encoders
for the SCARE architecture.

Pipeline:
1. LLM generates K diverse cooperation preferences (semantic behaviors)
2. LLM writes reward functions for each preference
3. Train K standard RNNAgents via QMIX, each with a different semantic reward
4. Save K checkpoints -> load into SCARERNNAgent's encoder bank
5. Generate alpha_prior (LLM tactical prior) for selector training

Usage:
    python scare_pretrain.py --env sc2 --n_encoders 4 --cuda_id 0
"""

import json
import os
import sys
# Skip -dataVersion when launching SC2 to avoid NGDP:E_NOT_AVAILABLE in offline environments
import re
import datetime
import argparse
import subprocess
import time
import textwrap
import numpy as np

# Add parent paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PYMARL_SRC = _THIS_DIR  # scare_pretrain.py is already in pymarl/src/
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from call_llm import LLM


# ============================================================
# Prompt templates (adapted from SemDiv's prompt_text_sc2.py)
# ============================================================

PROMPT_ENV_SC2 = (
    'You are an expert in cooperative multi-agent reinforcement learning (MARL) '
    'and code generation. We are going to train a team of two players in the '
    'Starcraft Multi-Agent Challenge (SMAC) game, which involves unit '
    'micromanagement tasks. In this game, ally units need to beat enemy units '
    'controlled by the built-in AI. Specifically, each player controls a marine '
    'agent ("1" and "2") to beat four enemy marines ("A", "B", "C", and "D"). '
    'The two marine agents are spawned at the center of the field, and four '
    'enemies are scattered in four different corners. Agents need to choose a '
    'same enemy, move towards it, and fire at it together to kill it. When '
    'agents successfully kill the first enemy, like enemy "B", they get a '
    'reward about 10 and the game ends. If both agents are killed, they lose.\n'
)

PROMPT_CODE_SC2 = '''
Here's a part of the original code:

```python
class Game:
    self.agents_position : {"1": np.ndarray[(2,)], "2": np.ndarray[(2,)]}
    self.enemies_position : {"A": np.ndarray[(2,)], "B": np.ndarray[(2,)], "C": np.ndarray[(2,)], "D": np.ndarray[(2,)]}
    self.killed_enemy : str

    def agent_enemy_distance(self, agent_idx: str, enemy_idx: str):
        agent_pos = self.agents_position[agent_idx]
        enemy_pos = self.enemies_position[enemy_idx]
        distance = np.linalg.norm(agent_pos - enemy_pos)
        return distance

    def step(self):
        reward = 0.0
        reward += self.additional_reward()
```
'''

PROMPT_MULTI_MODALITY = (
    "\nHuman player teams may have specific cooperation preferences to play "
    "the game, like attacking different enemies. They have their own "
    "`additional_reward` shown in the code. A new player outside a team needs "
    "to learn and adapt these preferences to cooperate well after joining the team.\n"
)

PROMPT_WRITE_CODE = '''
According to this cooperation preference, write an operational and executable reward function that formats as 'def additional_reward(self) -> float' and returns the 'reward : float' only.

1. Please think step by step and tell us what this code means;
2. The code function must align with the cooperation preference.
3. It can be a dense reward that guides the team to learn the cooperation preference.
4. Short and simple code is better.
'''

# 3m (SMAC v1): 3 marines vs 3 marines
PROMPT_ENV_SC2_3M = (
    'You are an expert in cooperative multi-agent reinforcement learning (MARL) '
    'and code generation. We are going to train a team of three players in the '
    'Starcraft Multi-Agent Challenge (SMAC) game, which involves unit '
    'micromanagement tasks. In this game, ally units need to beat enemy units '
    'controlled by the built-in AI. Specifically, each player controls a marine '
    'agent ("1", "2", and "3") to beat three enemy marines ("A", "B", and "C"). '
    'The three marine agents are spawned on one side of the field, and three '
    'enemies are spawned on the other side. Agents need to coordinate their '
    'fire to focus on the same enemy and kill it. When agents successfully kill '
    'an enemy, they get a reward about 10. If all enemies are killed, the team '
    'wins with a bonus reward of 200. If all agents are killed, they lose.\n'
)

PROMPT_CODE_SC2_3M = '''
Here's a part of the original code:

```python
class Game:
    self.agents_position : {"1": np.ndarray[(2,)], "2": np.ndarray[(2,)], "3": np.ndarray[(2,)]}
    self.enemies_position : {"A": np.ndarray[(2,)], "B": np.ndarray[(2,)], "C": np.ndarray[(2,)]}
    self.killed_enemy : str

    def agent_enemy_distance(self, agent_idx: str, enemy_idx: str):
        agent_pos = self.agents_position[agent_idx]
        enemy_pos = self.enemies_position[enemy_idx]
        distance = np.linalg.norm(agent_pos - enemy_pos)
        return distance

    def step(self):
        reward = 0.0
        reward += self.additional_reward()
```
'''

# ============================================================
# Auto-generate prompts from SMAC map metadata
# ============================================================

# Inline copy of map_param_registry to avoid import-time dependency on smac/pysc2
_MAP_PARAM_REGISTRY = {
    "3m": {"n_agents": 3, "n_enemies": 3, "map_type": "marines", "a_race": "T", "b_race": "T"},
    "8m": {"n_agents": 8, "n_enemies": 8, "map_type": "marines", "a_race": "T", "b_race": "T"},
    "25m": {"n_agents": 25, "n_enemies": 25, "map_type": "marines", "a_race": "T", "b_race": "T"},
    "5m_vs_6m": {"n_agents": 5, "n_enemies": 6, "map_type": "marines", "a_race": "T", "b_race": "T"},
    "8m_vs_9m": {"n_agents": 8, "n_enemies": 9, "map_type": "marines", "a_race": "T", "b_race": "T"},
    "10m_vs_11m": {"n_agents": 10, "n_enemies": 11, "map_type": "marines", "a_race": "T", "b_race": "T"},
    "27m_vs_30m": {"n_agents": 27, "n_enemies": 30, "map_type": "marines", "a_race": "T", "b_race": "T"},
    "MMM": {"n_agents": 10, "n_enemies": 10, "map_type": "MMM", "a_race": "T", "b_race": "T"},
    "MMM2": {"n_agents": 10, "n_enemies": 12, "map_type": "MMM", "a_race": "T", "b_race": "T"},
    "2s3z": {"n_agents": 5, "n_enemies": 5, "map_type": "stalkers_and_zealots", "a_race": "P", "b_race": "P"},
    "3s5z": {"n_agents": 8, "n_enemies": 8, "map_type": "stalkers_and_zealots", "a_race": "P", "b_race": "P"},
    "3s5z_vs_3s6z": {"n_agents": 8, "n_enemies": 9, "map_type": "stalkers_and_zealots", "a_race": "P", "b_race": "P"},
    "3s_vs_3z": {"n_agents": 3, "n_enemies": 3, "map_type": "stalkers", "a_race": "P", "b_race": "P"},
    "3s_vs_4z": {"n_agents": 3, "n_enemies": 4, "map_type": "stalkers", "a_race": "P", "b_race": "P"},
    "3s_vs_5z": {"n_agents": 3, "n_enemies": 5, "map_type": "stalkers", "a_race": "P", "b_race": "P"},
    "1c3s5z": {"n_agents": 9, "n_enemies": 9, "map_type": "colossi_stalkers_zealots", "a_race": "P", "b_race": "P"},
    "2m_vs_1z": {"n_agents": 2, "n_enemies": 1, "map_type": "marines", "a_race": "T", "b_race": "P"},
    "corridor": {"n_agents": 6, "n_enemies": 24, "map_type": "zealots", "a_race": "P", "b_race": "Z"},
    "6h_vs_8z": {"n_agents": 6, "n_enemies": 8, "map_type": "hydralisks", "a_race": "Z", "b_race": "P"},
    "2s_vs_1sc": {"n_agents": 2, "n_enemies": 1, "map_type": "stalkers", "a_race": "P", "b_race": "Z"},
    "so_many_baneling": {"n_agents": 7, "n_enemies": 32, "map_type": "zealots", "a_race": "P", "b_race": "Z"},
    "bane_vs_bane": {"n_agents": 24, "n_enemies": 24, "map_type": "bane", "a_race": "Z", "b_race": "Z"},
    "2c_vs_64zg": {"n_agents": 2, "n_enemies": 64, "map_type": "colossus", "a_race": "P", "b_race": "Z"},
}


def _parse_composition(map_name, unit_chars):
    """Parse unit composition from map name like '3s5z' or '1c3s5z'.

    unit_chars: dict mapping single char -> unit name, e.g. {'s': 'stalkers', 'z': 'zealots'}
    Returns human-readable string like '3 stalkers and 5 zealots'.
    """
    # Irregular plurals for SC2 units
    _singular = {'colossi': 'colossus', 'zealots': 'zealot', 'stalkers': 'stalker',
                 'marines': 'marine', 'zerglings': 'zergling', 'hydralisks': 'hydralisk'}
    parts = []
    for match in re.finditer(r'(\d+)([a-zA-Z])', map_name):
        count = int(match.group(1))
        char = match.group(2).lower()
        if char in unit_chars:
            name = unit_chars[char]
            # Singularize if count == 1
            if count == 1:
                name = _singular.get(name, name.rstrip('s'))
            parts.append(f"{count} {name}")
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _get_unit_descriptions(map_name, map_type, params):
    """Return (ally_description, enemy_description) strings for the map."""
    if map_type == "marines":
        ally = "marines"
        # 2m_vs_1z: marines vs zealot
        if params["b_race"] == "P":
            enemy = "zealots" if params["n_enemies"] > 1 else "zealot"
        else:
            enemy = "marines"
        return ally, enemy

    if map_type == "stalkers_and_zealots":
        unit_chars = {'s': 'stalkers', 'z': 'zealots'}
        # Parse ally side from map_name (before _vs_ if present)
        ally_part = map_name.split("_vs_")[0]
        ally = _parse_composition(ally_part, unit_chars) or "stalkers and zealots"
        # Parse enemy side
        if "_vs_" in map_name:
            enemy_part = map_name.split("_vs_")[1]
            enemy = _parse_composition(enemy_part, unit_chars) or ally
        else:
            enemy = ally
        return ally, enemy

    if map_type == "stalkers":
        ally = "stalkers"
        if params["b_race"] == "Z":
            enemy = "spine crawlers" if params["n_enemies"] == 1 else "spine crawlers"
        else:
            enemy = "zealots" if params["n_enemies"] > 1 else "zealot"
        return ally, enemy

    if map_type == "MMM":
        desc = "marines, marauders, and medivacs"
        return desc, desc

    if map_type == "hydralisks":
        return "hydralisks", "zealots"

    if map_type == "zealots":
        ally = "zealots"
        if params["b_race"] == "Z":
            enemy = "banelings and zerglings"
        else:
            enemy = "zealots"
        return ally, enemy

    if map_type == "bane":
        desc = "banelings and zerglings"
        return desc, desc

    if map_type == "colossus":
        ally = "colossi" if params["n_agents"] > 1 else "colossus"
        enemy = "zerglings"
        return ally, enemy

    if map_type == "colossi_stalkers_zealots":
        unit_chars = {'c': 'colossi', 's': 'stalkers', 'z': 'zealots'}
        ally_part = map_name.split("_vs_")[0]
        ally = _parse_composition(ally_part, unit_chars) or "colossi, stalkers, and zealots"
        if "_vs_" in map_name:
            enemy_part = map_name.split("_vs_")[1]
            enemy = _parse_composition(enemy_part, unit_chars) or ally
        else:
            enemy = ally
        return ally, enemy

    # Fallback
    return "units", "units"


def _make_enemy_labels(n_enemies):
    """Generate enemy labels: A, B, C, ... for <=26, else E1, E2, ..."""
    if n_enemies <= 26:
        return [chr(64 + i) for i in range(1, n_enemies + 1)]
    return [f"E{i}" for i in range(1, n_enemies + 1)]


def _format_labels(labels):
    """Format label list as quoted comma-separated string with 'and'."""
    quoted = [f'"{l}"' for l in labels]
    if len(quoted) <= 2:
        return " and ".join(quoted)
    return ", ".join(quoted[:-1]) + ", and " + quoted[-1]


def build_prompts_for_map(map_name):
    """Auto-generate LLM prompts from SMAC map metadata.

    Returns dict with 'env' and 'code' keys containing prompt strings.
    Falls back to the inline registry so smac/pysc2 need not be installed.
    """
    if map_name in _MAP_PARAM_REGISTRY:
        params = _MAP_PARAM_REGISTRY[map_name]
    else:
        # Try importing from smac at runtime
        try:
            from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
            registry = get_smac_map_registry()
            if map_name not in registry:
                raise ValueError(f"Unknown SMAC map: '{map_name}'")
            params = registry[map_name]
        except ImportError:
            raise ValueError(
                f"Unknown map '{map_name}' and smac package not available for lookup"
            )

    n_agents = params["n_agents"]
    n_enemies = params["n_enemies"]
    map_type = params["map_type"]

    ally_desc, enemy_desc = _get_unit_descriptions(map_name, map_type, params)

    agent_labels = [str(i + 1) for i in range(n_agents)]
    enemy_labels = _make_enemy_labels(n_enemies)

    # For composite descriptions like "3 stalkers and 5 zealots", skip the redundant total count
    ally_has_counts = any(c.isdigit() for c in ally_desc)
    enemy_has_counts = any(c.isdigit() for c in enemy_desc)
    ally_phrase = ally_desc if ally_has_counts else f'{n_agents} {ally_desc}'
    enemy_phrase = f'{enemy_desc}' if enemy_has_counts else f'{n_enemies} enemy {enemy_desc}'

    prompt_env = (
        f'You are an expert in cooperative multi-agent reinforcement learning (MARL) '
        f'and code generation. We are going to train a team of {n_agents} players in the '
        f'Starcraft Multi-Agent Challenge (SMAC) game "{map_name}", which involves unit '
        f'micromanagement tasks. In this game, ally units need to beat enemy units '
        f'controlled by the built-in AI. Specifically, the team controls '
        f'{ally_phrase} (agents {_format_labels(agent_labels)}) to beat '
        f'{enemy_phrase} '
        f'(enemies {_format_labels(enemy_labels)}). '
        f'Agents need to coordinate their attacks to focus fire and eliminate enemies. '
        f'When agents kill an enemy, they get a reward about 10. If all enemies are killed, '
        f'the team wins with a bonus reward of 200. If all agents are killed, they lose.\n'
    )

    agents_pos_type = ", ".join(f'"{l}": np.ndarray[(2,)]' for l in agent_labels)
    enemies_pos_type = ", ".join(f'"{l}": np.ndarray[(2,)]' for l in enemy_labels)

    prompt_code = f'''
Here's a part of the original code:

```python
class Game:
    self.agents_position : {{{agents_pos_type}}}
    self.enemies_position : {{{enemies_pos_type}}}
    self.killed_enemy : str

    def agent_enemy_distance(self, agent_idx: str, enemy_idx: str):
        agent_pos = self.agents_position[agent_idx]
        enemy_pos = self.enemies_position[enemy_idx]
        distance = np.linalg.norm(agent_pos - enemy_pos)
        return distance

    def step(self):
        reward = 0.0
        reward += self.additional_reward()
```
'''
    return {'env': prompt_env, 'code': prompt_code}


# Environment-specific prompt registry
# 'sc2' keeps the original hardcoded 2v4 prompt; 'sc2_v1' is now generated dynamically
ENV_PROMPTS = {
    'sc2': {
        'env': PROMPT_ENV_SC2,
        'code': PROMPT_CODE_SC2,
    },
    # sc2_v1 is populated dynamically via build_prompts_for_map() in SCAREPretrainer.__init__
}


class SCAREPretrainer:
    """
    Pre-trains K semantically diverse encoders using LLM-generated reward functions.

    Follows SemDiv's pipeline:
    1. generate_behavior() -> LLM proposes a cooperation preference
    2. generate_reward_code() -> LLM writes additional_reward function
    3. train_encoder() -> Train standard QMIX RNNAgent with semantic reward
    4. Repeat K times to build the encoder bank
    5. generate_alpha_prior() -> LLM generates tactical prior for selector
    """

    def __init__(self, env, env_file=None, n_encoders=4, cuda_id=0, map_name="3m"):
        self.llm = LLM(mode='openai')
        self.big_model = True
        self.env = env
        self.map_name = map_name
        self.n_encoders = n_encoders
        self.cuda_id = cuda_id

        # Default env_file for sc2/sc2_v1
        if env_file is None and 'sc2' in env:
            env_file = os.path.join(_THIS_DIR, 'envs', 'sc2_scare_reward.py')
        self.env_file = env_file

        self.timing = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.lib_dir = os.path.join('scare_lib', f'{self.timing}_scare')
        os.makedirs(self.lib_dir, exist_ok=True)

        # Libraries to track progress
        self.behavior_library = {}   # behavior_idx -> {behavior, code, status, ...}
        self.encoder_library = {}    # encoder_idx -> {behavior, code, model_path}

        self.max_attempt_per_behavior = 3
        self.max_behavior_total = 30

        # Validation gate parameters
        self.performance_tolerance = 0.2      # original task performance tolerance
        self.additional_reward_threshold = 0.1 # additional reward improvement threshold
        self.do_traj_check = True             # whether to do trajectory alignment check
        self.do_similarity_check = True       # whether to do similarity check
        self.traj_check_n_votes = 3           # LLM voting rounds for trajectory alignment
        self.similarity_tolerance = 0.2       # similarity tolerance

        # Read original env file for restoration
        with open(self.env_file, 'r') as f:
            self.original_env_file = f.read()

        # Get env-specific prompts
        if env == 'sc2_v1':
            # Dynamically generate prompts from map metadata
            self.prompts = build_prompts_for_map(self.map_name)
        elif env in ENV_PROMPTS:
            self.prompts = ENV_PROMPTS[env]
        else:
            raise ValueError(f"Environment '{env}' not supported. Available: {list(ENV_PROMPTS.keys()) + ['sc2_v1']}")

    def run(self):
        """Main pipeline: generate K diverse encoders."""
        encoder_idx = 1
        behavior_idx = 0

        while encoder_idx <= self.n_encoders:
            behavior_idx += 1
            if behavior_idx > self.max_behavior_total:
                print(f"[SCARE] Reached max behavior attempts ({self.max_behavior_total}). "
                      f"Only trained {encoder_idx - 1}/{self.n_encoders} encoders.")
                break

            # Step 1: Generate a cooperation preference
            behavior = self._generate_behavior(behavior_idx, encoder_idx)
            if behavior is None:
                continue

            self.behavior_library[behavior_idx] = {
                'behavior': behavior,
                'attempt_history': {},
                'status': 'pending',
            }
            self._save_progress()

            # Step 2: Try to write reward code and train
            success = False
            for attempt in range(1, self.max_attempt_per_behavior + 1):
                print(f"[SCARE] Encoder {encoder_idx}/{self.n_encoders}, "
                      f"Behavior {behavior_idx}, Attempt {attempt}")

                code = self._generate_reward_code(behavior, behavior_idx, attempt)
                if code is None:
                    self.behavior_library[behavior_idx]['attempt_history'][attempt] = {
                        'status': 'bug', 'info': 'LLM failed to generate valid code'
                    }
                    continue

                self.behavior_library[behavior_idx]['attempt_history'][attempt] = {
                    'code': code, 'status': 'training'
                }
                self._save_progress()

                # Inject reward function into env file
                if not self._inject_reward_code(code, encoder_idx):
                    self.behavior_library[behavior_idx]['attempt_history'][attempt]['status'] = 'bug'
                    self._restore_env_file()
                    continue

                # Train standard QMIX agent
                model_path = self._train_encoder(encoder_idx)
                self._restore_env_file()

                # 6-gate validation
                status, info, validated_path = self._check_training_status(
                    encoder_idx, model_path, behavior)

                if status == 'success':
                    self.behavior_library[behavior_idx]['status'] = 'success'
                    self.encoder_library[encoder_idx] = {
                        'behavior': behavior,
                        'code': code,
                        'model_path': validated_path,
                        'metrics': info,
                    }
                    self._save_progress()
                    encoder_idx += 1
                    success = True
                    break
                else:
                    self.behavior_library[behavior_idx]['attempt_history'][attempt] = {
                        'code': code, 'status': status, 'info': info
                    }

            if not success:
                self.behavior_library[behavior_idx]['status'] = 'failed'
                self._save_progress()

        # Step 3: Generate alpha prior
        if len(self.encoder_library) > 0:
            self._generate_and_save_alpha_prior()

        print(f"[SCARE] Pre-training complete. {len(self.encoder_library)} encoders trained.")
        print(f"[SCARE] Results saved to: {self.lib_dir}")
        return self.encoder_library

    def _generate_behavior(self, behavior_idx, encoder_idx):
        """Use LLM to generate a cooperation preference."""
        prompt = self.prompts['env'] + self.prompts['code'] + PROMPT_MULTI_MODALITY

        if encoder_idx == 1:
            prompt += (
                "\nBased on the information above, think step by step to come up "
                "with a possible cooperation preference. The preference should be "
                "deterministic and concrete. It should be as simple as possible. "
                "Avoid conditional terms like if, unless, when, etc. "
                "It should not conflict with the original task objective.\n\n"
                "Finally, output the preference in the format: "
                "'Human players may prefer to {preference}'"
            )
        else:
            # Few-shot: show existing successful behaviors to encourage diversity
            prompt += "Here are cooperation preferences we already have:\n"
            for idx, enc_info in self.encoder_library.items():
                prompt += f" - Encoder {idx}: {enc_info['behavior']}\n"

            # Negative examples: show rejected 'similar' behaviors
            similar_behaviors = [
                b for b in self.behavior_library.values()
                if any(
                    h.get('status') == 'similar'
                    for h in b.get('attempt_history', {}).values()
                )
            ]
            if similar_behaviors:
                prompt += "\nThese preferences were tried but rejected for being too similar:\n"
                for b in similar_behaviors:
                    prompt += f" - {b['behavior']} (too similar to existing encoders)\n"

            prompt += (
                "\nBased on the information above, think step by step to come up "
                "with another DIFFERENT cooperation preference that is NOT similar "
                "to the existing ones. The preference should be deterministic and "
                "concrete. It should be as simple as possible.\n\n"
                "Finally, output the preference in the format: "
                "'Human players may prefer to {preference}'"
            )

        output = self.llm.call_llm(prompt, big_model=self.big_model)
        if output == 'No answer!':
            return None

        # Extract the preference
        behavior = self.llm.call_llm(
            f'Task: Find and extract the part describing the specific cooperation '
            f'preference from the original text, ignoring other analysis related '
            f'content. Output the cooperation preference only.\n\n'
            f'Input: ({output})\n\nOutput: ',
            big_model=False
        )
        print(f"[SCARE] Generated behavior: {behavior}")
        return behavior

    def _generate_reward_code(self, behavior, behavior_idx, attempt):
        """Use LLM to write an additional_reward function for the behavior."""
        prompt = (
            self.prompts['env'] + self.prompts['code'] +
            f"\nNow we want to train a team with this specific cooperation behavior:"
            f"\n---\n{behavior}\n---" + PROMPT_WRITE_CODE
        )

        # Add feedback from previous failed attempts
        if attempt > 1:
            prompt += "\nWe have tried some reward function code before, but they are not good enough:\n"
            for prev in range(1, attempt):
                hist = self.behavior_library[behavior_idx]['attempt_history'].get(prev, {})
                if 'code' in hist:
                    prompt += f"Attempt {prev}: [\n{hist['code']}\n]\n"
                attempt_status = hist.get('status', '')
                attempt_info = hist.get('info', '')
                if attempt_status == 'bug':
                    prompt += f"This code has bugs: {attempt_info}\n"
                elif attempt_status == 'failed':
                    prompt += f"The team trained with this reward failed the original task. Performance: {attempt_info}\n"
                elif attempt_status == 'constant':
                    prompt += f"The additional reward values stayed near {attempt_info}, meaning the team cannot optimize it.\n"
                elif attempt_status == 'misaligned':
                    reason = self.llm.call_llm(
                        f"Summarize why the behavior is misaligned in <30 words:\n{attempt_info}", big_model=False)
                    prompt += f"The trained behavior does not match the preference. Reason: [{reason}]\n"
                elif attempt_status == 'similar':
                    prompt += f"The trained encoder is too similar to existing ones:\n"
                    if isinstance(attempt_info, dict):
                        for enc_idx, enc_info in attempt_info.items():
                            if isinstance(enc_info, dict) and enc_info.get('is_similar'):
                                prompt += f" - Encoder {enc_idx}: {enc_info.get('behavior', '')}\n"
            prompt += "\nPlease try a different approach."

        try:
            code_output = self.llm.call_llm(prompt, big_model=self.big_model)
            if "def additional_reward(self" not in code_output:
                return None

            # Extract clean code
            code = self.llm.call_llm(
                code_output +
                '\n\nTODO: Extract the python function `additional_reward` in the text. '
                'Output the function only (start with "```python\\ndef", end with "\\n```") '
                'so that i can directly copy it into my code.',
                big_model=False
            )
            store_code = code.split("```python\n")[1].split('```')[0]
            # Rename to llm{encoder_idx}
            store_code = store_code.replace(
                "def additional_reward(self", f"def llm_encoder(self"
            )
            return store_code
        except Exception as e:
            print(f"[SCARE] Code generation failed: {e}")
            return None

    def _inject_reward_code(self, code, encoder_idx):
        """Inject the reward function into the environment file."""
        try:
            if self.env == 'sc2_v1':
                # For sc2_v1: append as AdditionalReward.llm{encoder_idx} method
                # The code already has `def llm_encoder(self` — rename to llm{encoder_idx}
                injected_code = code.replace(
                    "def llm_encoder(self", f"def llm{encoder_idx}(self"
                )
                write_code = (
                    f"\n    '''\n    SCARE Encoder {encoder_idx} - "
                    f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\n    '''\n"
                )
                # Indent as class method (4 spaces)
                for line in injected_code.strip().split('\n'):
                    write_code += f"    {line}\n"
            else:
                write_code = (
                    f"\n'''\nSCARE Encoder {encoder_idx} - "
                    f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\n'''\n{code}\n"
                )
                write_code = textwrap.indent(write_code, '    ')

            with open(self.env_file, 'a') as f:
                f.write(write_code)
                f.flush()
                os.fsync(f.fileno())

            # Verify
            with open(self.env_file, 'r') as f:
                content = f.read()
            if self.env == 'sc2_v1':
                return f'def llm{encoder_idx}(self' in content
            return 'def llm_encoder(self' in content
        except Exception as e:
            print(f"[SCARE] Injection failed: {e}")
            return False

    def _restore_env_file(self):
        """Restore the original environment file."""
        with open(self.env_file, 'w') as f:
            f.write(self.original_env_file)

    def _train_encoder(self, encoder_idx):
        """Train a standard QMIX RNNAgent with the injected semantic reward."""
        run_name = f'scare_encoder_{encoder_idx}'

        if 'sc2' in self.env:
            t_max = 10000000
            if self.env == 'sc2_v1':
                env_config = 'sc2_scare'
                extra_args = f'env_args.additional_reward_id={encoder_idx} env_args.map_name={self.map_name} '
            else:
                env_config = 'sc2_v2'
                extra_args = ''
            cmd = (
                f'python {_PYMARL_SRC}/main.py '
                f'--config=qmix --env-config={env_config} '
                f'with name={run_name} '
                f'{extra_args}'
                f't_max={t_max} '
                f'use_cuda=True '
                f'gpu_id={self.cuda_id}'
            )
        else:
            print(f"[SCARE] Environment {self.env} training not yet configured")
            return None

        print(f"[SCARE] Training encoder {encoder_idx}: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[SCARE] Training failed (exit code {result.returncode})")
            print(f"[SCARE] STDOUT (last 1000 chars): {result.stdout[-1000:]}")
            print(f"[SCARE] STDERR (last 1000 chars): {result.stderr[-1000:]}")
            return None

        # Find the saved model
        model_dir = os.path.join(
            os.path.dirname(_PYMARL_SRC), 'results', 'models'
        )
        if os.path.exists(model_dir):
            for model in sorted(os.listdir(model_dir), reverse=True):
                if run_name in model:
                    model_path = os.path.join(model_dir, model)
                    print(f"[SCARE] Encoder {encoder_idx} saved: {model_path}")
                    return model_path

        print(f"[SCARE] Could not find saved model for encoder {encoder_idx}")
        return None

    def _find_tb_log_dir(self, encoder_idx):
        """Find TensorBoard log directory for a given encoder run."""
        run_name = f'scare_encoder_{encoder_idx}'
        tb_base = os.path.join(os.path.dirname(_PYMARL_SRC), 'results', 'tb_logs')
        if not os.path.exists(tb_base):
            return None
        for d in sorted(os.listdir(tb_base), reverse=True):
            if run_name in d:
                return os.path.join(tb_base, d)
        return None

    def _read_tb_scalars(self, tb_dir, tags):
        """Read scalar values from TensorBoard logs.

        Returns dict of tag -> list of (step, value) tuples.
        """
        try:
            from tensorboard.backend.event_processing import event_accumulator
            ea = event_accumulator.EventAccumulator(tb_dir)
            ea.Reload()
            result = {}
            available = ea.Tags().get('scalars', [])
            for tag in tags:
                if tag in available:
                    events = ea.Scalars(tag)
                    result[tag] = [(e.step, e.value) for e in events]
                else:
                    result[tag] = []
            return result
        except Exception as e:
            print(f"[SCARE] TB read failed: {e}")
            return {tag: [] for tag in tags}

    def _check_training_status(self, encoder_idx, model_path, behavior):
        """6-gate validation pipeline for a trained encoder.

        Returns (status, info, model_path) where:
        - status: 'success' | 'bug' | 'failed' | 'constant' | 'misaligned' | 'similar'
        - info: diagnostic details for LLM feedback
        - model_path: validated path (or None)
        """
        run_name = f'scare_encoder_{encoder_idx}'

        # Gate 1: Result dir + model file existence
        if model_path is None or not os.path.exists(model_path):
            print(f"[SCARE] Gate 1 FAIL: model not found")
            return ('bug', 'Training did not produce a model checkpoint', None)

        # Gate 2: TB log existence + crash detection
        tb_dir = self._find_tb_log_dir(encoder_idx)
        if tb_dir is None:
            print(f"[SCARE] Gate 2 FAIL: no TB logs found")
            return ('bug', 'Training crashed — no TensorBoard logs found', None)

        tb_tags = ['test_return_mean', 'test_reward_original_mean',
                    'test_reward_additional_mean', 'test_battle_won_mean']
        tb_data = self._read_tb_scalars(tb_dir, tb_tags)

        if not tb_data.get('test_return_mean'):
            print(f"[SCARE] Gate 2 FAIL: TB logs empty (training likely crashed)")
            return ('bug', 'Training crashed — TensorBoard logs are empty', None)

        # Gate 3: Original task performance check
        original_vals = tb_data.get('test_reward_original_mean', [])
        return_vals = tb_data.get('test_return_mean', [])
        if original_vals:
            final_original = np.mean([v for _, v in original_vals[-5:]])
        elif return_vals:
            # Fallback: use total return as proxy
            final_original = np.mean([v for _, v in return_vals[-5:]])
        else:
            final_original = 0.0

        # Baseline: first few episodes' return (before additional reward kicks in)
        if return_vals:
            baseline = np.mean([v for _, v in return_vals[:3]]) if len(return_vals) >= 3 else return_vals[0][1]
            # Only check if baseline is meaningful (> 0)
            if baseline > 0 and final_original < baseline * (1 - self.performance_tolerance):
                print(f"[SCARE] Gate 3 FAIL: original performance {final_original:.2f} < "
                      f"baseline {baseline:.2f} * {1 - self.performance_tolerance}")
                return ('failed', f'Original performance {final_original:.2f} dropped below '
                        f'baseline {baseline:.2f} (tolerance {self.performance_tolerance})', model_path)

        # Gate 4: Additional reward improvement check
        additional_vals = tb_data.get('test_reward_additional_mean', [])
        if additional_vals and len(additional_vals) >= 3:
            early_additional = np.mean([v for _, v in additional_vals[:3]])
            late_additional = np.mean([v for _, v in additional_vals[-5:]])
            if abs(early_additional) < 1e-8:
                improvement = 0.0 if abs(late_additional) < 1e-8 else float('inf')
            else:
                improvement = (late_additional - early_additional) / (abs(early_additional) + 1e-8)
            if improvement < self.additional_reward_threshold:
                print(f"[SCARE] Gate 4 FAIL: additional reward improvement {improvement:.3f} "
                      f"< threshold {self.additional_reward_threshold}")
                return ('constant', f'{late_additional:.4f} (early: {early_additional:.4f}, '
                        f'improvement: {improvement:.1%})', model_path)

        # Gate 5: Trajectory alignment (LLM-as-judge)
        if self.do_traj_check:
            traj_result = self._check_trajectory_alignment(
                encoder_idx, behavior, tb_data, model_path)
            if traj_result is not None:
                print(f"[SCARE] Gate 5 FAIL: trajectory misaligned")
                return traj_result

        # Gate 6: Similarity check
        if self.do_similarity_check and len(self.encoder_library) > 0:
            sim_result = self._check_similarity(encoder_idx, tb_data)
            if sim_result is not None:
                print(f"[SCARE] Gate 6 FAIL: too similar to existing encoder")
                return sim_result

        # All gates passed
        metrics = {
            'final_return': float(np.mean([v for _, v in return_vals[-5:]])) if return_vals else 0.0,
            'final_original': float(final_original),
        }
        if additional_vals:
            metrics['final_additional'] = float(np.mean([v for _, v in additional_vals[-5:]]))
        won_vals = tb_data.get('test_battle_won_mean', [])
        if won_vals:
            metrics['final_win_rate'] = float(np.mean([v for _, v in won_vals[-5:]]))
        print(f"[SCARE] All 6 gates PASSED for encoder {encoder_idx}: {metrics}")
        return ('success', metrics, model_path)

    def _check_trajectory_alignment(self, encoder_idx, behavior, tb_data, model_path):
        """LLM-as-judge voting: does the trained behavior match the target preference?

        Uses TB statistics (win rate, return, additional reward) as a simplified
        proxy for trajectory data. 3-round majority vote.

        Returns ('misaligned', info) or None if aligned.
        """
        return_vals = tb_data.get('test_return_mean', [])
        won_vals = tb_data.get('test_battle_won_mean', [])
        additional_vals = tb_data.get('test_reward_additional_mean', [])

        final_return = np.mean([v for _, v in return_vals[-5:]]) if return_vals else 0.0
        final_win_rate = np.mean([v for _, v in won_vals[-5:]]) if won_vals else 0.0
        final_additional = np.mean([v for _, v in additional_vals[-5:]]) if additional_vals else 0.0

        judge_prompt = (
            f"{self.prompts['env']}\n"
            f"A team was trained with the following cooperation preference:\n"
            f'"{behavior}"\n\n'
            f"After training, the team's statistics are:\n"
            f"- Win rate: {final_win_rate:.2%}\n"
            f"- Average return: {final_return:.2f}\n"
            f"- Average additional reward (for the preference): {final_additional:.4f}\n\n"
            f"Based on these statistics, does the team's behavior likely align with "
            f"the cooperation preference described above?\n"
            f"Answer with ONLY 'YES' or 'NO'."
        )

        yes_votes = 0
        for _ in range(self.traj_check_n_votes):
            answer = self.llm.call_llm(judge_prompt, big_model=False).strip().upper()
            if 'YES' in answer:
                yes_votes += 1

        aligned = yes_votes > self.traj_check_n_votes // 2
        print(f"[SCARE] Trajectory alignment vote: {yes_votes}/{self.traj_check_n_votes} YES")

        if not aligned:
            return ('misaligned', judge_prompt, model_path)
        return None

    def _check_similarity(self, encoder_idx, tb_data):
        """Check if the new encoder is too similar to existing ones.

        Compares TB metrics (return, additional_reward) between the new encoder
        and all existing encoders in the library.

        Returns ('similar', info, model_path) or None if sufficiently different.
        """
        new_return_vals = tb_data.get('test_return_mean', [])
        new_additional_vals = tb_data.get('test_reward_additional_mean', [])
        new_return = np.mean([v for _, v in new_return_vals[-5:]]) if new_return_vals else 0.0
        new_additional = np.mean([v for _, v in new_additional_vals[-5:]]) if new_additional_vals else 0.0

        similar_info = {}
        for existing_idx, enc_info in self.encoder_library.items():
            existing_metrics = enc_info.get('metrics', {})
            existing_return = existing_metrics.get('final_return', 0.0)
            existing_additional = existing_metrics.get('final_additional', 0.0)

            return_diff = abs(new_return - existing_return) / (abs(existing_return) + 1e-8)
            additional_diff = abs(new_additional - existing_additional) / (abs(existing_additional) + 1e-8)

            is_similar = (return_diff < self.similarity_tolerance and
                          additional_diff < self.similarity_tolerance)
            if is_similar:
                similar_info[existing_idx] = {
                    'is_similar': True,
                    'behavior': enc_info['behavior'],
                    'return_diff': f'{return_diff:.2%}',
                    'additional_diff': f'{additional_diff:.2%}',
                }

        if similar_info:
            return ('similar', similar_info, None)
        return None

    def _generate_and_save_alpha_prior(self):
        """
        Use LLM to generate a tactical prior alpha_LLM for the selector.

        For each possible scenario/preference description, the LLM selects
        which encoder best matches, producing a soft distribution over K encoders.
        This is saved as a JSON file for use during SCARE training.
        """
        encoder_descriptions = {}
        for idx, enc_info in self.encoder_library.items():
            encoder_descriptions[idx] = enc_info['behavior']

        # Generate prior: for each encoder's behavior, ask LLM to rate relevance
        # This creates a K x K affinity matrix, then normalize rows to get prior
        K = len(encoder_descriptions)
        alpha_prior = {}

        for query_idx, query_behavior in encoder_descriptions.items():
            prompt = (
                f"You are an expert in cooperative MARL. "
                f"We have {K} trained cooperation policies:\n"
            )
            for idx, desc in encoder_descriptions.items():
                prompt += f"  Policy {idx}: {desc}\n"
            prompt += (
                f"\nA new teammate says: \"{query_behavior}\"\n"
                f"Rate how well each policy (1-{K}) matches this preference "
                f"on a scale of 0-10. Output as JSON: "
                f'{{\"1\": score, \"2\": score, ...}}'
            )

            output = self.llm.call_llm(prompt, big_model=self.big_model)
            try:
                # Extract JSON from output
                json_match = re.search(r'\{[^}]+\}', output)
                if json_match:
                    scores = json.loads(json_match.group())
                    # Normalize to probability distribution
                    total = sum(float(v) for v in scores.values())
                    if total > 0:
                        alpha_prior[query_idx] = {
                            k: float(v) / total for k, v in scores.items()
                        }
                    else:
                        alpha_prior[query_idx] = {
                            str(i): 1.0 / K for i in range(1, K + 1)
                        }
            except Exception as e:
                print(f"[SCARE] Alpha prior generation failed for encoder {query_idx}: {e}")
                alpha_prior[query_idx] = {
                    str(i): 1.0 / K for i in range(1, K + 1)
                }

        # Save
        prior_path = os.path.join(self.lib_dir, 'alpha_prior.json')
        with open(prior_path, 'w') as f:
            json.dump(alpha_prior, f, indent=2)
        print(f"[SCARE] Alpha prior saved to: {prior_path}")

    def _save_progress(self):
        """Save current progress to disk."""
        with open(os.path.join(self.lib_dir, 'behavior.json'), 'w') as f:
            json.dump(self.behavior_library, f, indent=2)
        with open(os.path.join(self.lib_dir, 'encoder.json'), 'w') as f:
            json.dump(self.encoder_library, f, indent=2)


def load_pretrained_encoders(scare_agent, lib_dir):
    """
    Load pre-trained RNNAgent checkpoints into SCARERNNAgent's encoder bank.

    Args:
        scare_agent: SCARERNNAgent instance
        lib_dir: Path to scare_lib directory containing encoder.json
    """
    import torch as th

    encoder_path = os.path.join(lib_dir, 'encoder.json')
    with open(encoder_path, 'r') as f:
        encoder_library = json.load(f)

    for idx_str, enc_info in encoder_library.items():
        idx = int(idx_str) - 1  # 0-indexed for encoder bank
        model_path = enc_info['model_path']
        agent_path = os.path.join(model_path, 'agent.th')

        if os.path.exists(agent_path):
            rnn_state_dict = th.load(agent_path, map_location='cpu')
            scare_agent.load_encoder_from_rnn(rnn_state_dict, encoder_idx=idx)
            print(f"[SCARE] Loaded encoder {idx} from: {agent_path}")
        else:
            print(f"[SCARE] WARNING: Checkpoint not found: {agent_path}")

    # Freeze all encoders after loading
    scare_agent.freeze_encoders()
    print(f"[SCARE] All encoders frozen. Loaded {len(encoder_library)} encoders.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SCARE Encoder Pre-training')
    parser.add_argument('--env', type=str, default='sc2', help='Environment name')
    parser.add_argument('--env_file', type=str, default=None,
                        help='Path to environment.py file to inject reward functions')
    parser.add_argument('--n_encoders', type=int, default=4, help='Number of encoders to train')
    parser.add_argument('--cuda_id', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--map_name', type=str, default='3m', help='SMAC map name (e.g. 3m, 3s5z, MMM2)')
    args = parser.parse_args()

    pretrainer = SCAREPretrainer(
        env=args.env,
        env_file=args.env_file,
        n_encoders=args.n_encoders,
        cuda_id=args.cuda_id,
        map_name=args.map_name,
    )
    pretrainer.run()
