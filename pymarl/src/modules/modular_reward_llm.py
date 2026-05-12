import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import requests


def _get_arg(args, primary, fallback, default):
    if args is None:
        return default
    if hasattr(args, primary):
        return getattr(args, primary)
    if hasattr(args, fallback):
        return getattr(args, fallback)
    return default


class ModularRewardLLMInitializer:
    def __init__(self, logger=None, args=None):
        self.logger = logger
        self.args = args
        self.model = _get_arg(args, "modular_reward_llm_model", "tactic_llm_model", "gpt-4o")
        self.temperature = _get_arg(args, "modular_reward_llm_temperature", "tactic_llm_temperature", 0.2)
        self.base_url = _get_arg(args, "modular_reward_llm_base_url", "tactic_llm_base_url", "https://api.chatanywhere.tech/v1/chat/completions")
        self.timeout = _get_arg(args, "modular_reward_llm_timeout", "tactic_llm_timeout", 60)
        self.api_keys = self._load_api_keys()
        self.env_code_snippets = self._load_env_code_snippets()

    def initialize(self, module_catalog: List[Dict], env_name: str, env_args: Dict, module_budget: int) -> Dict:
        prompt = self._build_initial_pool_prompt(module_catalog, env_name, env_args, module_budget)
        result = {
            "source": "default",
            "model": self.model,
            "prompt": prompt,
            "raw_response": "",
            "modules": [],
        }

        if not _get_arg(self.args, "modular_reward_use_llm_init", "tactic_use_llm_init", True):
            return result

        if not self.api_keys:
            self._log("Modular reward LLM init skipped: no API key available, using defaults.")
            return result

        try:
            raw_response = self._chat(prompt)
            result["raw_response"] = raw_response
            parsed = self._parse_generated_modules_response(raw_response, module_budget)
            if not parsed:
                parsed = self._parse_response(raw_response, module_catalog, module_budget)
            if parsed:
                result["source"] = self.model
                result["modules"] = parsed
                return result
            self._log("Modular reward LLM init returned unparsable content, using defaults.")
        except Exception as exc:
            self._log("Modular reward LLM init failed: {}. Using defaults.".format(exc))

        return result

    def refine_scales(self, current_specs: Dict, stats_payload: Dict) -> List[float]:
        if not _get_arg(self.args, "modular_reward_use_llm_refinement", "tactic_use_llm_refinement", False):
            return []
        if not self.api_keys:
            return []

        prompt = self._build_refinement_prompt(current_specs, stats_payload)
        try:
            raw_response = self._chat(prompt)
            refined_scales = self._parse_refined_scales(raw_response, current_specs)
            if refined_scales:
                return refined_scales
        except Exception as exc:
            self._log("Modular reward LLM refinement failed: {}. Keeping current scales.".format(exc))
        return []

    def propose_replacement(
        self,
        current_specs: Dict,
        stats_payload: Dict,
        env_name: str,
        env_args: Dict,
        interface_payload: Dict,
        target_module: Dict,
    ) -> Dict:
        prompt = self._build_replacement_prompt(
            current_specs=current_specs,
            stats_payload=stats_payload,
            env_name=env_name,
            env_args=env_args,
            interface_payload=interface_payload,
            target_module=target_module,
        )
        result = {
            "source": "default",
            "model": self.model,
            "prompt": prompt,
            "raw_response": "",
            "action": "keep_module",
            "proposal": None,
        }

        if not _get_arg(self.args, "modular_reward_use_llm_generation", "tactic_use_llm_generation", True):
            return result
        if not self.api_keys:
            self._log("Modular reward module generation skipped: no API key available.")
            return result

        try:
            raw_response = self._chat(prompt)
            result["raw_response"] = raw_response
            parsed = self._parse_replacement_response(raw_response)
            if parsed is not None:
                result["source"] = self.model
                result.update(parsed)
                return result
            self._log("Modular reward module generation returned unparsable content.")
        except Exception as exc:
            self._log("Modular reward module generation failed: {}.".format(exc))
        return result

    def _chat(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(random.choice(self.api_keys)),
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You design executable modular auxiliary reward modules for cooperative MARL. Return strict JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _map_facts(self, map_name: str) -> List[str]:
        key = str(map_name).lower()
        registry = {
            "3m": [
                "3m: 3 allied Marines vs 3 enemy Marines. Symmetric Terran composition, all ranged.",
            ],
            "8m": [
                "8m: 8 allied Marines vs 8 enemy Marines. Symmetric Terran composition, all ranged.",
            ],
            "25m": [
                "25m: 25 allied Marines vs 25 enemy Marines. Symmetric Terran composition, all ranged.",
            ],
            "5m_vs_6m": [
                "5m_vs_6m: 5 allied Marines vs 6 enemy Marines. Terran ranged mirror with a 1-unit numerical disadvantage for the ally.",
            ],
            "8m_vs_9m": [
                "8m_vs_9m: 8 allied Marines vs 9 enemy Marines. Terran ranged mirror with a 1-unit numerical disadvantage for the ally.",
            ],
            "10m_vs_11m": [
                "10m_vs_11m: 10 allied Marines vs 11 enemy Marines. Terran ranged mirror with a 1-unit numerical disadvantage for the ally.",
            ],
            "27m_vs_30m": [
                "27m_vs_30m: 27 allied Marines vs 30 enemy Marines. Terran ranged mirror with a 3-unit numerical disadvantage for the ally.",
            ],
            "mmm": [
                "MMM: ally has 1 Medivac, 2 Marauders, and 7 Marines; enemy has the same composition.",
                "Marines are basic ranged Terran infantry.",
                "Marauders are ranged Terran units with anti-armor damage and slower movement than Marines.",
                "Medivacs are Terran support air units that heal nearby allied biological units; they do not deal damage.",
            ],
            "mmm2": [
                "MMM2: ally has 1 Medivac, 2 Marauders, and 7 Marines; enemy has 1 Medivac, 3 Marauders, and 8 Marines (1 extra Marauder and 1 extra Marine).",
                "Marines are basic ranged Terran infantry.",
                "Marauders are ranged Terran units with anti-armor damage and slower movement than Marines.",
                "Medivacs are Terran support air units that heal nearby allied biological units; they do not deal damage.",
            ],
            "2s3z": [
                "2s3z: ally has 2 Stalkers and 3 Zealots; enemy has the same composition.",
                "Stalkers are ranged Protoss units.",
                "Zealots are melee Protoss units that must reach melee contact to deal damage.",
            ],
            "3s5z": [
                "3s5z: ally has 3 Stalkers and 5 Zealots; enemy has the same composition.",
                "Stalkers are ranged Protoss units.",
                "Zealots are melee Protoss units that must reach melee contact to deal damage.",
            ],
            "3s5z_vs_3s6z": [
                "3s5z_vs_3s6z: ally has 3 Stalkers and 5 Zealots; enemy has 3 Stalkers and 6 Zealots (1 extra Zealot).",
                "Stalkers are ranged Protoss units.",
                "Zealots are melee Protoss units that must reach melee contact to deal damage.",
            ],
            "3s_vs_3z": [
                "3s_vs_3z: 3 allied Stalkers vs 3 enemy Zealots. Equal unit count, asymmetric types.",
                "Stalkers are ranged Protoss units; Zealots are melee Protoss units that must reach melee contact to deal damage.",
            ],
            "3s_vs_4z": [
                "3s_vs_4z: 3 allied Stalkers vs 4 enemy Zealots (ally has a 1-unit numerical disadvantage).",
                "Stalkers are ranged Protoss units; Zealots are melee Protoss units that must reach melee contact to deal damage.",
            ],
            "3s_vs_5z": [
                "3s_vs_5z: 3 allied Stalkers vs 5 enemy Zealots (ally has a 2-unit numerical disadvantage).",
                "Stalkers are ranged Protoss units; Zealots are melee Protoss units that must reach melee contact to deal damage.",
            ],
            "2m_vs_1z": [
                "2m_vs_1z: 2 allied Marines (ranged Terran) vs 1 enemy Zealot (melee Protoss, must reach melee contact to deal damage).",
            ],
            "2s_vs_1sc": [
                "2s_vs_1sc: 2 allied Stalkers (ranged Protoss) vs 1 enemy Spine Crawler (stationary ranged Zerg defensive unit).",
            ],
            "1c3s5z": [
                "1c3s5z: ally has 1 Colossus, 3 Stalkers, and 5 Zealots; enemy has the same composition.",
                "Colossi are high-HP ranged Protoss units with area-of-effect attacks.",
                "Stalkers are ranged Protoss units.",
                "Zealots are melee Protoss units that must reach melee contact to deal damage.",
            ],
            "2c_vs_64zg": [
                "2c_vs_64zg: 2 allied Colossi (ranged Protoss with area-of-effect attack) vs 64 enemy Zerglings (fast melee Zerg swarm).",
            ],
            "corridor": [
                "corridor: 6 allied Zealots (melee Protoss) vs 24 enemy Zerglings (fast melee Zerg swarm), with the engagement taking place in a narrow corridor-shaped layout.",
            ],
            "6h_vs_8z": [
                "6h_vs_8z: 6 allied Hydralisks (ranged Zerg) vs 8 enemy Zealots (melee Protoss). Ally has a 2-unit numerical disadvantage.",
            ],
            "bane_vs_bane": [
                "bane_vs_bane: ally has 20 Zerglings and 4 Banelings; enemy has the same composition.",
                "Zerglings are fast melee Zerg units.",
                "Banelings are Zerg units with area-of-effect damage that die when they explode on contact.",
            ],
            "so_many_baneling": [
                "so_many_baneling: 7 allied Zealots (melee Protoss) vs 32 enemy Banelings (Zerg area-of-effect suicide units that die when they explode on contact).",
            ],
        }
        return registry.get(key, [])

    def _proxy_feature_semantics(self) -> List[str]:
        return [
            "Per-agent proxy signal definitions (plain facts, no strategy implied):",
            "- features['attack_avail']: 1 if at least one attack action is currently available to this agent, else 0.",
            "- features['chose_attack']: 1 if this agent selected an attack action this step, else 0.",
            "- features['chose_move']: 1 if this agent selected a movement action this step, else 0.",
            "- features['chose_idle']: 1 if this agent selected the no-op/idle action this step, else 0.",
            "- features['has_teammate_same_target']: 1 if this agent attacks and at least one teammate attacks the same enemy this step, else 0.",
            "- features['same_target_as_prev']: 1 if this agent attacked both the previous and current step with the same target index, else 0.",
            "- features['switched_target']: 1 if this agent attacked both the previous and current step but the target index changed, else 0.",
            "- features['same_action_as_prev']: 1 if this agent's action index is identical at the previous and current step, else 0.",
            "- features['team_attack_ratio']: fraction of allied agents that selected an attack action this step.",
            "- features['same_target_ratio']: fraction of attacking allies this step whose target equals this agent's target.",
            "- tensors['actions']: integer tensor [batch, time, n_agents, 1] of chosen action indices.",
            "- tensors['avail_actions']: 0/1 tensor [batch, time, n_agents, n_actions] of which actions are available.",
            "- tensors['attack_action_start']: the smallest action index that represents 'attack enemy k'; indices below this are no-op / stop / movement.",
        ]

    def _shaping_aggregation_note(self) -> List[str]:
        return [
            "Shaping aggregation contract (facts about how module outputs are consumed):",
            "- each module outputs a per-agent tensor r_i[batch, time, n_agents].",
            "- the selector combines per-agent module outputs into per-agent contributions c_i[batch, time, n_agents].",
            "- the team shaped-reward delta added to the environment reward is beta * mean_over_agents(sum_of_selected c_i).",
            "- the same per-agent contributions also drive a per-agent auxiliary Q loss that back-propagates directly into each agent's network.",
        ]

    def _build_initial_pool_prompt(self, module_catalog: List[Dict], env_name: str, env_args: Dict, module_budget: int) -> str:
        map_name = env_args.get("map_name", "unknown")
        lines = [
            "We are building the initial modular reward agent for cooperative MARL in PyMARL.",
            "The environment is {} on map {}.".format(env_name, map_name),
            "Task: generate a small executable pool of auxiliary reward modules.",
            "Do not just pick from a catalog; generate the actual reward module implementations.",
            "You are asked to propose modules; you are not given any predetermined strategy and must reason from the facts below.",
            "Return JSON with this schema:",
            '{"modules": [{"name": "module_name", "description": "short text", "when_to_use": "short text", "required_inputs": ["..."], "hypothesis": "short text", "expected_effect": "short text", "python_function_source": "def compute_reward_module(features, tensors):\\n    ..."}]}',
            "Return exactly {} modules unless the interface constraints make one impossible.".format(module_budget),
            "Each module must belong to a meaningfully different behaviour family; avoid near-duplicates that only rename the same heuristic.",
            "Behaviour families available in the proxy feature interface include: focus_fire (teammates attacking the same target), target_persistence (keeping the same target across steps), target_handoff (switching target when appropriate), attack_commitment (attacking rather than idling), team_participation (fraction of team simultaneously attacking), and action_adaptation (alignment or change in chosen action across steps). Prefer covering distinct families over stacking variants of the same family.",
            "Treat these families as orthogonal slots. If two candidate modules would end up using essentially the same underlying signal (or one is the sign-flipped / ratio-vs-boolean version of another), drop one and fill the slot with an underrepresented family instead.",
            "Non-linear shaping (e.g. a U-shape, clipped gap, or interaction between two features) is preferred over a single raw feature whenever it captures the intent more faithfully.",
        ]
        if str(map_name).lower() == "3s_vs_5z":
            lines.extend([
                "3s_vs_5z is 3 allied ranged Stalkers against 5 enemy melee Zealots; allies have a 2-unit numerical disadvantage but a range advantage.",
                "Because the interface does not expose positions, health, or distances, do not invent kiting-distance, hp-threshold, or melee-contact tensors; express the map-specific prior only through the allowed attack, idle, same-target, switched-target, same-action, team-attack, and same-target-ratio proxies.",
                "For 3s_vs_5z, prefer modules that improve concentrated damage on a single Zealot so the ranged team removes melee attackers one at a time, discourage idle frames during kiting-style combat, and stabilize target persistence so Zealots cannot get free hits from split damage.",
            ])
        if str(map_name).upper() == "MMM2":
            lines.extend([
                "MMM2 is a heterogeneous Terran mirror with Marines, Marauders, and Medivacs, so prioritize modules that improve coordinated pressure under mixed-unit combat rather than only simple 3m-style target focus.",
                "Because the runtime interface here does not expose unit types, medivac identity, or raw health vectors, do not invent healer-specific or unit-type-specific tensors; express MMM2 knowledge only through the allowed proxy coordination signals.",
                "For MMM2, prefer modules that separate roles such as synchronized target focus, attack commitment, anti-idle pressure, team attack participation balance, adaptive target handoff, and late-fight target synchronization.",
            ])
        elif str(map_name).upper() == "27M_VS_30M":
            lines.extend([
                "27m_vs_30m is a homogeneous Marine-vs-Marine attrition map with a numeric disadvantage, so prioritize modules that improve disciplined focus fire, synchronized attack participation, anti-idle pressure, and target persistence under losing trades rather than heterogeneous-role reasoning.",
                "Because both sides are pure Marines and the interface does not expose health arrays or positions, do not invent kiting-distance, hp-threshold, or unit-type-specific tensors; express the map-specific prior only through the allowed attack, idle, same-target, and team-participation proxies.",
                "For 27m_vs_30m, prefer modules that help the team avoid scattered fire, reduce wasted hesitation, maintain attack commitment once pressure starts, and stabilize coordinated target handoff during extended mirror fights.",
            ])
        map_facts = self._map_facts(map_name)
        if map_facts:
            lines.append("Map facts (objective composition only, no strategy):")
            for fact in map_facts:
                lines.append("- {}".format(fact))
        lines.append("Winning condition: eliminate all enemy units. Losing condition: all allied units die or the episode times out.")
        lines.extend(self._proxy_feature_semantics())
        lines.extend(self._shaping_aggregation_note())
        lines.extend([
            "Allowed observation / action interface comes from the environment snippets below.",
            "Environment snippets:",
        ])
        for snippet in self.env_code_snippets:
            lines.append("```python")
            lines.append(snippet)
            lines.append("```")
        lines.extend([
            "Strict interface contract:",
            "- function signature must be: def compute_reward_module(features, tensors):",
            "- you may only reference these feature keys: attack_avail, chose_attack, chose_move, chose_idle, has_teammate_same_target, same_target_as_prev, switched_target, same_action_as_prev, team_attack_ratio, same_target_ratio",
            "- you may only reference these tensor keys: actions, avail_actions, attack_action_start",
            "- do not invent keys like positions, health arrays, map boundaries, ally locations, enemy locations, or custom metadata",
            "- a preloaded torch-like namespace is already available as th and torch; do NOT write import torch or any import statement",
            "- return a torch tensor with the same shape as features['attack_avail']",
            "- prefer vectorized tensor expressions; avoid Python loops and list comprehensions",
            "Valid example:",
            "def compute_reward_module(features, tensors):\n    return features['has_teammate_same_target'] * features['attack_avail']",
            "Safety constraints:",
            "- no imports",
            "- no filesystem or network access",
            "- no subprocess or global state access",
            "- return finite torch tensors only",
            "Return JSON only. No markdown fences.",
        ])
        return "\n".join(lines)

    def _build_prompt(self, module_catalog: List[Dict], env_name: str, env_args: Dict, module_budget: int) -> str:
        map_name = env_args.get("map_name", "unknown")
        lines = [
            "We are building an LLM-guided modular reward agent for cooperative MARL in PyMARL.",
            "The environment is {} on map {}.".format(env_name, map_name),
            "Goal: choose a small executable pool of auxiliary reward modules for sparse/delayed team rewards.",
            "The modules must be selected only from the catalog below.",
            "Return JSON with this schema:",
            '{"modules": [{"name": "module_name", "description": "short text", "when_to_use": "short text"}]}',
            "Select at most {} modules. Prefer diverse modules with complementary incentives.".format(module_budget),
            "Catalog:",
        ]
        for module in module_catalog:
            lines.append("- {name}: {description}".format(**module))
        lines.extend([
            "Current design intent:",
            "- shared module pool",
            "- per-agent contextual selector",
            "- training reward = env reward + beta * auxiliary reward",
            "- choose modules that are useful for SMAC micromanagement and coordination",
            "Return JSON only. No markdown fences.",
        ])
        return "\n".join(lines)

    def _parse_generated_modules_response(self, text: str, module_budget: int) -> List[Dict]:
        payload = self._extract_json_object(text)
        if not isinstance(payload, dict):
            return []
        modules = payload.get("modules", [])
        if not isinstance(modules, list):
            return []

        parsed = []
        seen = set()
        for item in modules:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            source = str(item.get("python_function_source", "")).strip()
            if not name or not source or name in seen:
                continue
            seen.add(name)
            required_inputs = item.get("required_inputs", [])
            if not isinstance(required_inputs, list):
                required_inputs = []
            parsed.append({
                "name": name,
                "scale": 1.0,
                "description": str(item.get("description", "")),
                "when_to_use": str(item.get("when_to_use", "")),
                "required_inputs": [str(v).strip() for v in required_inputs if str(v).strip()],
                "hypothesis": str(item.get("hypothesis", "")),
                "expected_effect": str(item.get("expected_effect", "")),
                "python_function_source": source,
                "source_type": "generated",
            })
            if len(parsed) >= module_budget:
                break
        return parsed

    def _module_family_label(self, module: Dict) -> str:
        text = " ".join([
            str(module.get("name", "")),
            str(module.get("description", "")),
            str(module.get("when_to_use", "")),
            str(module.get("hypothesis", "")),
            str(module.get("expected_effect", "")),
        ]).lower()
        if any(token in text for token in ["focus_fire", "focus fire", "same target", "same-target"]):
            return "focus_fire"
        if any(token in text for token in ["target_persistence", "target persistence", "anti-switch", "switched_target", "switch target", "target handoff"]):
            return "target_control"
        if any(token in text for token in ["attack_commitment", "attack commitment", "attack participation", "sustained pressure", "attack_avail"]):
            return "attack_commitment"
        if any(token in text for token in ["coordinated_advance", "advance", "anti-idle", "idle", "close distance", "phase transition"]):
            return "advance_transition"
        if any(token in text for token in ["team_attack_balance", "balance", "participation balance", "distribution"]):
            return "team_balance"
        if any(token in text for token in ["action_variation", "variation", "adaptation", "diversity"]):
            return "action_adaptation"
        return "other"

    def _family_display_name(self, family: str) -> str:
        return {
            "focus_fire": "focus-fire / same-target",
            "target_control": "target persistence / handoff",
            "attack_commitment": "attack commitment / pressure",
            "advance_transition": "advance / anti-idle / phase-transition",
            "team_balance": "team attack balance",
            "action_adaptation": "action adaptation / variation",
            "other": "other",
        }.get(family, family)

    def _build_failure_tags(self, current_specs: Dict, stats_payload: Dict, target_module: Dict) -> List[str]:
        history = stats_payload.get("module_window_history", {}).get(target_module.get("id"), [])
        recent = history[-3:] if history else []
        tags = []
        if recent:
            mean_usage = sum(float(item.get("usage", 0.0)) for item in recent) / float(len(recent))
            mean_contribution = sum(float(item.get("contribution", 0.0)) for item in recent) / float(len(recent))
            mean_activation = sum(float(item.get("activation", 0.0)) for item in recent) / float(len(recent))
            if mean_usage <= 0.12:
                tags.append("inactive_module: selector almost never routes to this module")
            if mean_contribution <= 0.0:
                tags.append("low_contribution_module: weighted reward contribution is persistently weak or non-positive")
            if mean_activation <= 0.10:
                tags.append("low_activation_module: the module rarely exceeds meaningful routing weight")
            if mean_usage <= 0.20 and mean_activation <= 0.20 and mean_contribution <= 0.01:
                tags.append("late_game_ineffective: even when available, the module does not become meaningfully useful")

        selector_entropy = float(stats_payload.get("selector_entropy", 0.0))
        top1_switch_rate = float(stats_payload.get("selector_top1_switch_rate", 0.0))
        if selector_entropy <= 0.9 and top1_switch_rate <= 0.08:
            tags.append("selector_collapse_context: routing is becoming narrow, so replacements should encourage more state-conditional specialization")

        target_family = self._module_family_label(target_module)
        same_family_count = sum(
            1
            for module in current_specs.get("modules", [])
            if module.get("id") != target_module.get("id") and self._module_family_label(module) == target_family
        )
        if same_family_count > 0:
            tags.append(
                "redundant_family_risk: the current pool already contains {} other {} module(s)".format(
                    same_family_count,
                    self._family_display_name(target_family),
                )
            )

        if not tags:
            tags.append("weak_module: replace with a more distinct and useful module")
        return tags

    def _summarize_family_coverage(self, current_specs: Dict, target_module: Dict) -> List[str]:
        counts = {}
        for module in current_specs.get("modules", []):
            if module.get("id") == target_module.get("id"):
                continue
            family = self._module_family_label(module)
            counts[family] = counts.get(family, 0) + 1
        if not counts:
            return ["No other module families are currently active in the pool."]
        lines = []
        for family, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append("{}: {} module(s) already in pool".format(self._family_display_name(family), count))
        return lines

    def _suggest_functional_gaps(self, current_specs: Dict, target_module: Dict, env_args: Dict) -> List[str]:
        families = [
            self._module_family_label(module)
            for module in current_specs.get("modules", [])
            if module.get("id") != target_module.get("id")
        ]
        gaps = []
        if families.count("focus_fire") + families.count("target_control") >= max(2, len(families) // 2 if families else 0):
            gaps.append("Prefer an orthogonal module instead of another same-target / anti-switch variant.")
        if "advance_transition" not in families:
            gaps.append("A module for non-attack to attack phase transition or anti-idle advance is still under-covered.")
        if "attack_commitment" not in families:
            gaps.append("A module for sustained attack participation or pressure commitment is still under-covered.")
        if "team_balance" not in families:
            gaps.append("A module for team attack participation balance or synchronized pressure is still under-covered.")
        if not gaps:
            gaps.append("Prefer a module that fills the least represented coordination role in the current pool.")
        return gaps[:4]

    def _build_refinement_prompt(self, current_specs: Dict, stats_payload: Dict) -> str:
        lines = [
            "We are refining the scales of an existing modular reward agent.",
            "Keep the same module names and only propose updated scales.",
            "Return JSON with schema: {\"scales\": [float, ...]}",
            "Current modules:",
        ]
        for module in current_specs.get("modules", []):
            module_key = module.get("id", module.get("name"))
            usage = stats_payload.get("module_usage", {}).get(module_key, stats_payload.get("module_usage", {}).get(module.get("name"), 0.0))
            score = stats_payload.get("module_scores", {}).get(module_key, stats_payload.get("module_scores", {}).get(module.get("name"), 0.0))
            contribution = stats_payload.get("module_contributions", {}).get(module_key, stats_payload.get("module_contributions", {}).get(module.get("name"), 0.0))
            weighted_score = stats_payload.get("module_weighted_scores", {}).get(module_key, stats_payload.get("module_weighted_scores", {}).get(module.get("name"), 0.0))
            activation = stats_payload.get("module_activation", {}).get(module_key, stats_payload.get("module_activation", {}).get(module.get("name"), 0.0))
            lines.append(
                "- {name}: usage={usage:.4f}, score={score:.4f}, contribution={contribution:.4f}, weighted_score={weighted_score:.4f}, activation={activation:.4f}".format(
                    name=module.get("name"),
                    usage=usage,
                    score=score,
                    contribution=contribution,
                    weighted_score=weighted_score,
                    activation=activation,
                )
            )
        lines.extend([
            "Global stats:",
            "- latest_test_return={:.4f}".format(float(stats_payload.get("latest_test_return", 0.0))),
            "- selector_entropy={:.4f}".format(float(stats_payload.get("selector_entropy", 0.0))),
            "- aux_reward_mean={:.4f}".format(float(stats_payload.get("aux_reward_mean", 0.0))),
            "Rules:",
            "- Keep scales in [0.0, 2.0]",
            "- Be conservative: small changes only",
            "- Prefer reducing modules with low activation and weak contribution",
            "- Prefer keeping or slightly increasing modules with stable positive contribution",
            "Return JSON only.",
        ])
        return "\n".join(lines)

    def _build_replacement_prompt(
        self,
        current_specs: Dict,
        stats_payload: Dict,
        env_name: str,
        env_args: Dict,
        interface_payload: Dict,
        target_module: Dict,
    ) -> str:
        map_name = env_args.get("map_name", "unknown")
        feature_desc = interface_payload.get("feature_descriptions", {})
        tensor_desc = interface_payload.get("tensor_descriptions", {})
        module_history = stats_payload.get("module_window_history", {})
        target_history = module_history.get(target_module.get("id"), [])
        target_name = target_module.get("name", "unknown")
        failure_tags = self._build_failure_tags(current_specs, stats_payload, target_module)
        family_coverage = self._summarize_family_coverage(current_specs, target_module)
        functional_gaps = self._suggest_functional_gaps(current_specs, target_module, env_args)
        target_family = self._family_display_name(self._module_family_label(target_module))
        family_cap = int(getattr(self.args, "modular_reward_max_modules_per_family", 2)) if self.args is not None else 2

        lines = [
            "We are updating one auxiliary reward module in a PyMARL cooperative MARL system.",
            "Task: design a replacement executable reward module for {} / {} given only the facts and proxies below; do not rely on any predetermined strategy.".format(env_name, map_name),
            "Only update a module that has been persistently low-usage, low-contribution, or strategically redundant.",
            "Generate a replacement executable reward function module, not a scale tweak.",
            "The replacement should respond to the failure diagnosis below, not just rename the old heuristic.",
            "Target module family: {}.".format(target_family),
            "Failure summary tags:",
        ]
        for tag in failure_tags:
            lines.append("- {}".format(tag))
        map_facts = self._map_facts(map_name)
        if map_facts:
            lines.append("Map facts (objective composition only, no strategy):")
            for fact in map_facts:
                lines.append("- {}".format(fact))
        lines.append("Winning condition: eliminate all enemy units. Losing condition: all allied units die or the episode times out.")
        lines.extend(self._proxy_feature_semantics())
        lines.extend(self._shaping_aggregation_note())
        lines.extend([
            "Current family coverage in the module pool:",
        ])
        for item in family_coverage:
            lines.append("- {}".format(item))
        lines.extend([
            "Functional gaps to prefer in this replacement:",
        ])
        for item in functional_gaps:
            lines.append("- {}".format(item))
        lines.extend([
            "Environment code snippets:",
        ])
        for snippet in self.env_code_snippets:
            lines.append("```python")
            lines.append(snippet)
            lines.append("```")

        lines.extend([
            "Reward module execution interface:",
            "- function signature: {}".format(interface_payload.get("function_signature", "def compute_reward_module(features, tensors):")),
            "- return contract: {}".format(interface_payload.get("return_contract", "Return a torch tensor with shape [batch, time, n_agents].")),
            "Available feature tensors:",
        ])
        for name, description in feature_desc.items():
            lines.append("- features[{!r}]: {}".format(name, description))
        lines.append("Available raw tensors:")
        for name, description in tensor_desc.items():
            lines.append("- tensors[{!r}]: {}".format(name, description))

        lines.extend([
            "Current module pool:",
        ])
        for module in current_specs.get("modules", []):
            lines.append(
                "- {name} | id={id} | source_type={source_type} | status={status} | description={description}".format(
                    name=module.get("name"),
                    id=module.get("id"),
                    source_type=module.get("source_type", "builtin"),
                    status=module.get("status", "active"),
                    description=module.get("description", ""),
                )
            )
            source = str(module.get("python_function_source", "")).strip()
            if source:
                lines.append("  source preview: {}".format(self._trim_code(source, 280)))

        lines.extend([
            "Target module to replace:",
            "- name={name}, id={id}, source_type={source_type}, status={status}, family={family}".format(
                name=target_module.get("name"),
                id=target_module.get("id"),
                source_type=target_module.get("source_type", "builtin"),
                status=target_module.get("status", "active"),
                family=target_family,
            ),
            "- description={}".format(target_module.get("description", "")),
            "- when_to_use={}".format(target_module.get("when_to_use", "")),
            "Target module recent windows:",
        ])
        if target_history:
            for item in target_history[-8:]:
                lines.append(
                    "- t_env={t_env}, usage={usage:.4f}, contribution={contribution:.4f}, activation={activation:.4f}, weighted_score={weighted_score:.4f}".format(
                        t_env=int(item.get("t_env", 0)),
                        usage=float(item.get("usage", 0.0)),
                        contribution=float(item.get("contribution", 0.0)),
                        activation=float(item.get("activation", 0.0)),
                        weighted_score=float(item.get("weighted_score", 0.0)),
                    )
                )
        else:
            lines.append("- no history available")

        lines.extend([
            "Recent global training stats:",
            "- latest_test_return={:.4f}".format(float(stats_payload.get("latest_test_return", 0.0))),
            "- latest_test_win_rate={:.4f}".format(float(stats_payload.get("latest_test_win_rate", 0.0))),
            "- latest_train_return={:.4f}".format(float(stats_payload.get("latest_train_return", 0.0))),
            "- selector_entropy={:.4f}".format(float(stats_payload.get("selector_entropy", 0.0))),
            "- selector_advantage={:.4f}".format(float(stats_payload.get("selector_advantage", 0.0))),
            "- aux_reward_mean={:.4f}".format(float(stats_payload.get("aux_reward_mean", 0.0))),
            "Constraints:",
            "- Do not import anything.",
            "- Do not access filesystem, network, subprocesses, globals, or external state.",
            "- Use only the provided features/tensors and torch operations.",
            "- th and torch are already preloaded; do not write import torch.",
            "- Only allowed feature keys: attack_avail, chose_attack, chose_move, chose_idle, has_teammate_same_target, same_target_as_prev, switched_target, same_action_as_prev, team_attack_ratio, same_target_ratio.",
            "- Only allowed tensor keys: actions, avail_actions, attack_action_start.",
            "- Do not invent keys like positions, health arrays, enemy lists, map boundaries, or ally locations.",
            "- Return a dense tensor with the exact same shape as features['attack_avail'].",
            "- Keep outputs numerically stable and bounded; avoid NaN/Inf.",
            "- Do not produce another module whose main logic is still focus-fire, anti-switch, or generic attack encouragement unless it clearly fills a different functional gap than the existing pool.",
            "- If the target module is weak because it is redundant, replace it with a more orthogonal module family instead of a small variation of the same family.",
            "- The replacement must fill a distinct functional gap rather than just lightly renaming an existing heuristic.",
            "- Prefer replacements that improve mid-game or late-game coordination, not only early aggression or exploration.",
            "- Prefer modules that combine signals in a way that changes routing behavior, not modules that merely rescale an existing reward pattern.",
            "- Treat {} module(s) per family as a soft upper bound for the pool; if that family is already full, move to a different family.".format(family_cap),
            "Decision contract:",
            "Return strict JSON only with this schema:",
            '{"action":"replace_module|keep_module","module":{"name":"new_name","description":"short text","when_to_use":"short text","python_function_source":"def compute_reward_module(features, tensors):\\n    ...","required_inputs":["feature_or_tensor_name"],"hypothesis":"short text","expected_effect":"short text","parent_id":"%s","version_parent":%s}}' % (target_module.get("id", ""), int(target_module.get("version", 1))),
            "If the current module should remain unchanged, return {\"action\": \"keep_module\"}.",
        ])
        return "\n".join(lines)

    def _parse_refined_scales(self, text: str, current_specs: Dict) -> List[float]:
        payload = None
        candidates = [text]
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            candidates.insert(0, match.group(0))
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
                break
            except Exception:
                continue
        if not isinstance(payload, dict):
            return []
        scales = payload.get("scales", [])
        if not isinstance(scales, list) or len(scales) != len(current_specs.get("modules", [])):
            return []
        parsed = []
        for scale in scales:
            try:
                parsed.append(max(0.0, min(float(scale), 2.0)))
            except Exception:
                return []
        return parsed

    def _parse_response(self, text: str, module_catalog: List[Dict], module_budget: int) -> List[Dict]:
        catalog = {item["name"]: item for item in module_catalog}
        payload = None
        candidates = [text]
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            candidates.insert(0, match.group(0))

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
                break
            except Exception:
                continue

        if not isinstance(payload, dict):
            return []

        modules = payload.get("modules", [])
        if not isinstance(modules, list):
            return []

        parsed = []
        seen = set()
        for item in modules:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if name not in catalog or name in seen:
                continue
            seen.add(name)
            scale = item.get("scale", 1.0)
            try:
                scale = float(scale)
            except Exception:
                scale = 1.0
            parsed.append({
                "name": name,
                "scale": max(0.0, min(scale, 2.0)),
                "description": str(item.get("description", catalog[name]["description"])),
                "when_to_use": str(item.get("when_to_use", "")),
            })
            if len(parsed) >= module_budget:
                break
        return parsed

    def _parse_replacement_response(self, text: str) -> Optional[Dict]:
        payload = self._extract_json_object(text)
        if not isinstance(payload, dict):
            return None

        action = str(payload.get("action", "replace_module")).strip() or "replace_module"
        if action not in {"replace_module", "keep_module", "deprecate_module", "modify_module", "create_module"}:
            return None
        if action in {"keep_module", "deprecate_module"}:
            return {"action": action, "proposal": None}
        if action in {"modify_module", "create_module"}:
            action = "replace_module"

        module = payload.get("module") or payload.get("proposal")
        if not isinstance(module, dict):
            return None

        name = str(module.get("name", "")).strip()
        source = str(module.get("python_function_source", "")).strip()
        if not name or not source:
            return None

        required_inputs = module.get("required_inputs", [])
        if not isinstance(required_inputs, list):
            required_inputs = []
        required_inputs = [str(item).strip() for item in required_inputs if str(item).strip()]

        return {
            "action": action,
            "proposal": {
                "name": name,
                "scale": 1.0,
                "description": str(module.get("description", "")).strip(),
                "when_to_use": str(module.get("when_to_use", "")).strip(),
                "python_function_source": source,
                "required_inputs": required_inputs,
                "hypothesis": str(module.get("hypothesis", "")).strip(),
                "expected_effect": str(module.get("expected_effect", "")).strip(),
                "parent_id": str(module.get("parent_id", "")).strip(),
                "version_parent": int(module.get("version_parent", 1) or 1),
            },
        }

    def _extract_json_object(self, text: str):
        candidates = [text]
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            candidates.insert(0, match.group(0))
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return None

    def _load_api_keys(self) -> List[str]:
        keys = []
        for env_name in ["MODULAR_REWARD_OPENAI_API_KEY", "TACTIC_OPENAI_API_KEY", "OPENAI_API_KEY"]:
            value = os.environ.get(env_name, "").strip()
            if value:
                keys.append(value)

        source_path = _get_arg(self.args, "modular_reward_llm_key_source", "tactic_llm_key_source", "")
        if not source_path:
            source_path = "/root/ycw/human_aicoord/SemDiv/language/call_llm.py"

        if os.path.exists(source_path):
            try:
                with open(source_path, "r", encoding="utf-8") as handle:
                    content = handle.read()
                keys.extend(re.findall(r"sk-[A-Za-z0-9]+", content))
            except Exception:
                pass

        deduped = []
        seen = set()
        for key in keys:
            if key and key not in seen:
                seen.add(key)
                deduped.append(key)
        return deduped

    def _load_env_code_snippets(self) -> List[str]:
        snippets = []
        base_dir = Path(__file__).resolve().parent.parent
        candidates = [
            base_dir / "envs" / "multiagentenv.py",
            base_dir / "envs" / "sc2_v2_wrapper.py",
        ]
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    content = handle.read().strip()
                if content:
                    snippets.append(self._trim_code(content, 900))
            except Exception:
                continue
        return snippets

    def _trim_code(self, text: str, max_chars: int) -> str:
        stripped = "\n".join(line.rstrip() for line in text.strip().splitlines())
        if len(stripped) <= max_chars:
            return stripped
        return stripped[: max_chars - 3] + "..."

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.console_logger.info(message)
