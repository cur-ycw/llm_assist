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

    def _build_initial_pool_prompt(self, module_catalog: List[Dict], env_name: str, env_args: Dict, module_budget: int) -> str:
        map_name = env_args.get("map_name", "unknown")
        lines = [
            "We are building the initial modular reward agent for cooperative MARL in PyMARL.",
            "The environment is {} on map {}.".format(env_name, map_name),
            "Task: generate a small executable pool of auxiliary reward modules for sparse/delayed team rewards.",
            "Do not just pick from a catalog; generate the actual reward module implementations.",
            "Return JSON with this schema:",
            '{"modules": [{"name": "module_name", "description": "short text", "when_to_use": "short text", "required_inputs": ["..."], "hypothesis": "short text", "expected_effect": "short text", "python_function_source": "def compute_reward_module(features, tensors):\\n    ..."}]}',
            "Return exactly {} modules unless the interface constraints make one impossible.".format(module_budget),
            "Each module must belong to a meaningfully different behaviour family; avoid near-duplicates that only rename the same heuristic.",
            "Prefer coverage across distinct coordination roles such as: focus fire, attack commitment, target persistence / handoff, action adaptation, team attack balance, and no-attack advance / anti-idle pressure.",
            "At least half of the modules should target mid-game or late-game coordination weaknesses rather than early exploration only.",
            "If one family is already represented, the next modules should preferentially cover underrepresented coordination gaps instead of making minor variants of the same family.",
        ]
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
        lines.extend([
            "Allowed observation / action interface comes from the environment snippets below.",
            "Environment snippets:",
        ])
        for snippet in self.env_code_snippets:
            lines.append("```python")
            lines.append(snippet)
            lines.append("```")
        lines.extend([
            "Current design intent:",
            "- shared module pool",
            "- per-agent contextual selector",
            "- training reward = env reward + beta * auxiliary reward",
            "- choose modules that are useful for SMAC micromanagement and coordination",
            "- the module pool should contain complementary modules rather than repeated variants of the same heuristic",
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
        if str(env_args.get("map_name", "")).upper() == "MMM2":
            gaps.append("For MMM2, prefer mixed-fight coordination proxies like target handoff, sustained pressure, or phase-sensitive participation rather than another generic focus-fire heuristic.")
        elif str(env_args.get("map_name", "")).upper() == "27M_VS_30M":
            gaps.append("For 27m_vs_30m, prefer homogeneous-mirror coordination proxies like disciplined same-target pressure, anti-idle attack commitment, and stable target persistence rather than heterogeneous-role heuristics.")
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
            "Task: improve late-game coordination on {} / {} without destabilising training.".format(env_name, map_name),
            "Only update a module that has been persistently low-usage, low-contribution, or strategically redundant.",
            "Generate a replacement executable reward function module, not a scale tweak.",
            "The replacement should respond to the failure diagnosis below, not just rename the old heuristic.",
            "Target module family: {}.".format(target_family),
            "Failure summary tags:",
        ]
        if str(map_name).upper() == "MMM2":
            lines.extend([
                "MMM2 is a heterogeneous Marine/Marauder/Medivac fight, so replacements should improve mixed-team coordination proxies such as synchronized pressure, disciplined target handoff, and sustained attack participation.",
                "Do not assume access to medivac identity, ally health arrays, enemy health arrays, or unit-type masks; use only the allowed proxy coordination signals.",
                "Avoid proposing another generic focus-fire clone unless it uses a genuinely different signal combination than the current pool.",
            ])
        elif str(map_name).upper() == "27M_VS_30M":
            lines.extend([
                "27m_vs_30m is a homogeneous Marine mirror under unit-count disadvantage, so replacements should improve disciplined same-target damage concentration, sustained team attack participation, anti-idle pressure, and target persistence in long mirror skirmishes.",
                "Do not assume access to unit hp, positions, or spacing; use only the allowed proxy coordination signals.",
                "Avoid proposals that depend on heterogeneous-unit logic, healer logic, or distance-threshold logic that the runtime interface cannot observe.",
            ])
        for tag in failure_tags:
            lines.append("- {}".format(tag))
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
