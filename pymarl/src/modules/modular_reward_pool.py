import ast
import json
import math
import os
from copy import deepcopy
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch as th
import torch.nn.functional as F

from .modular_reward_llm import ModularRewardLLMInitializer


STATUS_TO_CODE = {
    "active": 1.0,
    "candidate": 0.5,
    "frozen": 0.0,
    "deprecated": -1.0,
}

ALLOWED_TORCH_FUNCTIONS = {
    "abs",
    "arange",
    "cat",
    "clamp",
    "exp",
    "gather",
    "log",
    "maximum",
    "mean",
    "minimum",
    "nan_to_num",
    "ones_like",
    "relu",
    "sigmoid",
    "softmax",
    "sqrt",
    "stack",
    "sum",
    "tensor",
    "tanh",
    "where",
    "zeros_like",
}

FORBIDDEN_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.Lambda,
    ast.ClassDef,
    ast.AsyncFunctionDef,
)

FORBIDDEN_CALL_NAMES = {
    "eval",
    "exec",
    "open",
    "compile",
    "input",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "vars",
    "dir",
    "locals",
    "globals",
    "help",
    "type",
    "super",
    "memoryview",
}

FORBIDDEN_ATTRIBUTE_ROOTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "requests",
    "urllib",
    "builtins",
}

ALLOWED_TENSOR_KEYS = {
    "actions",
    "avail_actions",
    "attack_action_start",
}

ALLOWED_FEATURE_KEYS = {
    "attack_avail",
    "chose_attack",
    "chose_move",
    "chose_idle",
    "has_teammate_same_target",
    "same_target_as_prev",
    "switched_target",
    "same_action_as_prev",
    "team_attack_ratio",
    "same_target_ratio",
}

SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "range": range,
    "len": len,
    "float": float,
    "int": int,
}


BUILTIN_MODULE_LIBRARY = {
    "focus_fire": {
        "description": "Reward agents for attacking the same target as teammates when attack actions are available.",
        "when_to_use": "When allies can already attack and need coordination.",
        "python_function_source": "def compute_reward_module(features, tensors):\n    return features['has_teammate_same_target'] + 0.2 * features['chose_attack'] * (1.0 - features['has_teammate_same_target'])\n",
    },
    "attack_commitment": {
        "description": "Reward agents for taking attack actions when attack is available and penalise hesitation.",
        "when_to_use": "When an agent is in range but dithers.",
        "python_function_source": "def compute_reward_module(features, tensors):\n    return features['attack_avail'] * features['chose_attack'] - 0.5 * features['attack_avail'] * (1.0 - features['chose_attack'])\n",
    },
    "coordinated_advance": {
        "description": "Reward agents for moving instead of idling when no attack is currently available.",
        "when_to_use": "When agents should close distance rather than idle.",
        "python_function_source": "def compute_reward_module(features, tensors):\n    return (1.0 - features['attack_avail']) * features['chose_move'] - 0.25 * (1.0 - features['attack_avail']) * features['chose_idle']\n",
    },
    "target_persistence": {
        "description": "Reward agents for staying on the same target across nearby timesteps instead of thrashing.",
        "when_to_use": "When rapid target switching hurts damage concentration.",
        "python_function_source": "def compute_reward_module(features, tensors):\n    return 0.5 * features['same_target_as_prev'] - 0.1 * features['switched_target']\n",
    },
    "action_variation": {
        "description": "Reward agents for avoiding repeated identical actions when attack opportunities remain dynamic.",
        "when_to_use": "When agents become stuck repeating low-value actions.",
        "python_function_source": "def compute_reward_module(features, tensors):\n    return features['attack_avail'] * (1.0 - features['same_action_as_prev'])\n",
    },
    "team_attack_balance": {
        "description": "Reward attack participation while discouraging over-concentration on the same target pattern.",
        "when_to_use": "When some agents attack while others lag or overkill pressure emerges.",
        "python_function_source": "def compute_reward_module(features, tensors):\n    return features['attack_avail'] * features['team_attack_ratio'] * (1.0 - 0.5 * features['same_target_ratio'])\n",
    },
}


def _get_arg(args, primary, fallback, default):
    if hasattr(args, primary):
        return getattr(args, primary)
    if hasattr(args, fallback):
        return getattr(args, fallback)
    return default


class ModularRewardModulePool:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.attack_action_start = _get_arg(args, "modular_reward_attack_action_start", "tactic_attack_action_start", 6)
        self.module_catalog = [
            {
                "name": "focus_fire",
                "description": BUILTIN_MODULE_LIBRARY["focus_fire"]["description"],
            },
            {
                "name": "attack_commitment",
                "description": BUILTIN_MODULE_LIBRARY["attack_commitment"]["description"],
            },
            {
                "name": "coordinated_advance",
                "description": BUILTIN_MODULE_LIBRARY["coordinated_advance"]["description"],
            },
            {
                "name": "target_persistence",
                "description": BUILTIN_MODULE_LIBRARY["target_persistence"]["description"],
            },
            {
                "name": "action_variation",
                "description": BUILTIN_MODULE_LIBRARY["action_variation"]["description"],
            },
            {
                "name": "team_attack_balance",
                "description": BUILTIN_MODULE_LIBRARY["team_attack_balance"]["description"],
            },
        ]
        self.initializer = ModularRewardLLMInitializer(logger=logger, args=args)
        self.spec_bundle = self._init_specs()
        self.module_specs = self.spec_bundle["modules"]
        self.pending_update = None
        self.last_update_events = []
        self.last_trigger_decision = {"mode": "init", "t_env": 0}
        self.module_window_history = {spec["id"]: [] for spec in self.module_specs}
        self.feedback_window_history = []
        self.last_transition_progress = 1.0
        self.compiled_module_functions = {}
        self._refresh_module_cache()
        self.last_module_usage = {spec["id"]: 0.0 for spec in self.module_specs}
        self.last_module_scores = {spec["id"]: 0.0 for spec in self.module_specs}
        self.last_module_contributions = {spec["id"]: 0.0 for spec in self.module_specs}
        self.last_module_weighted_scores = {spec["id"]: 0.0 for spec in self.module_specs}
        self.last_module_activation = {spec["id"]: 0.0 for spec in self.module_specs}
        self.last_module_status = {spec["id"]: spec["status"] for spec in self.module_specs}
        self.last_module_status_code = {spec["id"]: STATUS_TO_CODE.get(spec["status"], 0.0) for spec in self.module_specs}
        self.last_module_scale = {spec["id"]: float(spec["scale"]) for spec in self.module_specs}

    def _init_specs(self) -> Dict:
        module_budget = _get_arg(self.args, "modular_reward_module_budget", "tactic_module_budget", 4)
        env_name = getattr(self.args, "env", "unknown")
        env_args = getattr(self.args, "env_args", {})
        bundle = self.initializer.initialize(self.module_catalog, env_name, env_args, module_budget)

        fallback_modules = [
            {
                "name": "focus_fire",
                "scale": 1.0,
                "description": BUILTIN_MODULE_LIBRARY["focus_fire"]["description"],
                "when_to_use": BUILTIN_MODULE_LIBRARY["focus_fire"]["when_to_use"],
            },
            {
                "name": "attack_commitment",
                "scale": 0.8,
                "description": BUILTIN_MODULE_LIBRARY["attack_commitment"]["description"],
                "when_to_use": BUILTIN_MODULE_LIBRARY["attack_commitment"]["when_to_use"],
            },
            {
                "name": "coordinated_advance",
                "scale": 0.5,
                "description": BUILTIN_MODULE_LIBRARY["coordinated_advance"]["description"],
                "when_to_use": BUILTIN_MODULE_LIBRARY["coordinated_advance"]["when_to_use"],
            },
            {
                "name": "target_persistence",
                "scale": 0.4,
                "description": BUILTIN_MODULE_LIBRARY["target_persistence"]["description"],
                "when_to_use": BUILTIN_MODULE_LIBRARY["target_persistence"]["when_to_use"],
            },
            {
                "name": "action_variation",
                "scale": 0.7,
                "description": BUILTIN_MODULE_LIBRARY["action_variation"]["description"],
                "when_to_use": BUILTIN_MODULE_LIBRARY["action_variation"]["when_to_use"],
            },
            {
                "name": "team_attack_balance",
                "scale": 0.9,
                "description": BUILTIN_MODULE_LIBRARY["team_attack_balance"]["description"],
                "when_to_use": BUILTIN_MODULE_LIBRARY["team_attack_balance"]["when_to_use"],
            },
        ][:module_budget]

        candidate_modules = bundle.get("modules", []) or []
        validated_modules = []
        rejected_initial_modules = []
        for index, raw_spec in enumerate(candidate_modules[:module_budget]):
            spec = self._normalise_spec(raw_spec, index)
            fn, report = self._compile_module_spec(spec)
            dry_run = {"ok": False, "reason": "static_validation_failed"}
            if report.get("ok", False):
                dry_run = self._runtime_dry_run_report(spec, fn)
            validation_report = {
                "static_validation": report,
                "dry_run": dry_run,
            }
            spec["validator_report"] = validation_report
            if report.get("ok", False) and dry_run.get("ok", False):
                validated_modules.append(spec)
            else:
                rejected_initial_modules.append({
                    "index": index,
                    "name": spec.get("name"),
                    "reason": validation_report,
                })

        while len(validated_modules) < module_budget:
            fallback_index = len(validated_modules)
            fallback_spec = self._normalise_spec(fallback_modules[fallback_index], fallback_index)
            fallback_spec["source_type"] = "builtin"
            fallback_spec["validator_report"] = {"ok": True, "reason": "builtin_fallback"}
            validated_modules.append(fallback_spec)

        bundle["modules"] = validated_modules[:module_budget]
        bundle["rejected_initial_modules"] = rejected_initial_modules
        if rejected_initial_modules and self.logger is not None:
            self.logger.console_logger.info(
                "Rejected {} invalid initial LLM modules; filled remaining slots with builtin fallback.".format(
                    len(rejected_initial_modules)
                )
            )
        return bundle

    def _normalise_spec(self, spec: Dict, index: int) -> Dict:
        module_name = str(spec["name"])
        builtin = BUILTIN_MODULE_LIBRARY.get(module_name, {})
        scale = max(0.0, min(float(spec.get("scale", 1.0)), 2.0))
        source_type = str(spec.get("source_type", "builtin" if module_name in BUILTIN_MODULE_LIBRARY else "generated"))
        python_function_source = str(spec.get("python_function_source", builtin.get("python_function_source", "")))
        required_inputs = spec.get("required_inputs", [])
        if not isinstance(required_inputs, list):
            required_inputs = []
        return {
            "id": spec.get("id", "module_{}".format(index)),
            "name": module_name,
            "version": int(spec.get("version", 1)),
            "status": str(spec.get("status", "active")),
            "target_status": str(spec.get("target_status", spec.get("status", "active"))),
            "scale": scale,
            "description": str(spec.get("description", builtin.get("description", ""))),
            "when_to_use": str(spec.get("when_to_use", builtin.get("when_to_use", ""))),
            "created_at_t_env": int(spec.get("created_at_t_env", 0)),
            "last_updated_t_env": int(spec.get("last_updated_t_env", 0)),
            "promotion_score": float(spec.get("promotion_score", 0.0)),
            "rollback_count": int(spec.get("rollback_count", 0)),
            "cooldown_until": int(spec.get("cooldown_until", 0)),
            "source_type": source_type,
            "parent_id": spec.get("parent_id"),
            "fallback_id": spec.get("fallback_id", spec.get("id", "module_{}".format(index)) if source_type == "builtin" else None),
            "required_inputs": [str(item) for item in required_inputs],
            "hypothesis": str(spec.get("hypothesis", "")),
            "expected_effect": str(spec.get("expected_effect", "")),
            "python_function_source": python_function_source,
            "validator_report": deepcopy(spec.get("validator_report")),
            "candidate_shadow_scale": float(spec.get("candidate_shadow_scale", _get_arg(self.args, "modular_reward_candidate_shadow_scale", "tactic_candidate_shadow_scale", 0.25))),
            "transition_alpha": float(spec.get("transition_alpha", 1.0 if str(spec.get("status", "active")) == "active" else 0.0)),
            "transition_start_t_env": int(spec.get("transition_start_t_env", 0)),
        }

    def _refresh_module_cache(self):
        self.module_scales = th.tensor([spec["scale"] for spec in self.module_specs], dtype=th.float32)
        self.compiled_module_functions = {}
        for spec in self.module_specs:
            fn, report = self._compile_module_spec(spec)
            previous_report = deepcopy(spec.get("validator_report")) if isinstance(spec.get("validator_report"), dict) else None
            if previous_report and "static_validation" in previous_report:
                if report is not None:
                    previous_report["static_validation"] = report
                spec["validator_report"] = previous_report
            elif report is not None:
                spec["validator_report"] = report
            self.compiled_module_functions[spec["id"]] = fn
        self._sync_runtime_metadata()

    def _sample_runtime_tensors(self) -> Tuple[th.Tensor, th.Tensor]:
        n_actions = max(self.attack_action_start + 4, self.attack_action_start + 1)
        batch_size = 2
        time_steps = 5
        n_agents = 3
        sample_actions = th.randint(low=0, high=n_actions, size=(batch_size, time_steps, n_agents, 1))
        sample_avail = th.ones(batch_size, time_steps, n_agents, n_actions)
        return sample_actions, sample_avail

    def _runtime_dry_run_report(self, spec: Dict, fn) -> Dict:
        sample_actions, sample_avail = self._sample_runtime_tensors()
        features = self._build_features(sample_actions, sample_avail)
        tensors = self._build_execution_tensors(sample_actions, sample_avail)
        try:
            output = self._execute_module_function(spec, fn, features, tensors)
            return {
                "ok": True,
                "shape": list(output.shape),
                "mean": float(output.mean().item()),
                "min": float(output.min().item()),
                "max": float(output.max().item()),
            }
        except Exception as exc:
            return {"ok": False, "reason": "dry_run_failed", "detail": str(exc)}

    def save_specs(self, unique_token: str, local_results_path: str) -> None:
        base_dir = os.path.join(local_results_path, "modular_reward_modules")
        os.makedirs(base_dir, exist_ok=True)
        payload = dict(self.spec_bundle)
        payload["modules"] = [deepcopy(spec) for spec in self.module_specs]
        payload["module_names"] = [spec["name"] for spec in self.module_specs]
        payload["attack_action_start"] = self.attack_action_start
        payload["pending_update"] = self.pending_update
        payload["last_update_events"] = self.last_update_events
        payload["module_window_history"] = deepcopy(self.module_window_history)
        payload["feedback_window_history"] = deepcopy(self.feedback_window_history)
        payload["last_trigger_decision"] = deepcopy(self.last_trigger_decision)
        payload["rejected_initial_modules"] = deepcopy(self.spec_bundle.get("rejected_initial_modules", []))
        save_path = os.path.join(base_dir, "{}.json".format(unique_token))
        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        self.logger.console_logger.info("Saved modular reward specs to {}".format(save_path))

    def num_modules(self) -> int:
        return len(self.module_specs)

    def selector_input_dim(self) -> int:
        return 7

    @property
    def module_names(self) -> List[str]:
        return [spec["name"] for spec in self.module_specs]

    def update_scales(self, new_scales, t_env: int = 0, source: str = "manual"):
        if len(new_scales) != len(self.module_specs):
            raise ValueError("Scale count does not match module count")
        for spec, scale in zip(self.module_specs, new_scales):
            spec["scale"] = max(0.0, min(float(scale), 2.0))
            spec["version"] += 1
            spec["last_updated_t_env"] = int(t_env)
            spec["target_status"] = spec["status"]
        self._refresh_module_cache()
        self.last_update_events = [{
            "type": "scale_update",
            "source": source,
            "t_env": int(t_env),
        }]

    def get_specs_payload(self):
        return {
            "modules": [deepcopy(spec) for spec in self.module_specs],
            "module_names": self.module_names,
            "pending_update": deepcopy(self.pending_update),
        }

    def get_interface_payload(self) -> Dict:
        return {
            "function_signature": "def compute_reward_module(features, tensors):",
            "return_contract": "Return a torch tensor with shape [batch, time, n_agents] and finite values only.",
            "feature_descriptions": {
                "attack_avail": "Float tensor [batch, time, n_agents], 1 when any attack action is available.",
                "chose_attack": "Float tensor [batch, time, n_agents], 1 when the chosen action is an attack.",
                "chose_move": "Float tensor [batch, time, n_agents], 1 when the chosen action is a movement action.",
                "chose_idle": "Float tensor [batch, time, n_agents], 1 when the chosen action is no-op or stop.",
                "has_teammate_same_target": "Float tensor [batch, time, n_agents], 1 when another ally attacks the same target.",
                "same_target_as_prev": "Float tensor [batch, time, n_agents], 1 when the agent keeps the previous attack target.",
                "switched_target": "Float tensor [batch, time, n_agents], 1 when the agent changes attack target from previous step.",
                "same_action_as_prev": "Float tensor [batch, time, n_agents], 1 when the action repeats the previous action.",
                "team_attack_ratio": "Float tensor [batch, time, n_agents], fraction of allied agents attacking at that timestep.",
                "same_target_ratio": "Float tensor [batch, time, n_agents], same-target count normalized by number of agents.",
            },
            "tensor_descriptions": {
                "actions": "Long tensor [batch, time, n_agents, 1] with chosen action ids.",
                "avail_actions": "Float/int tensor [batch, time, n_agents, n_actions] with action availability mask.",
                "attack_action_start": "Integer index where attack actions begin.",
            },
        }

    def get_latest_stats(self):
        return {
            "module_usage": dict(self.last_module_usage),
            "module_scores": dict(self.last_module_scores),
            "module_contributions": dict(self.last_module_contributions),
            "module_weighted_scores": dict(self.last_module_weighted_scores),
            "module_activation": dict(self.last_module_activation),
            "module_status": dict(self.last_module_status),
            "module_status_code": dict(self.last_module_status_code),
            "module_scale": dict(self.last_module_scale),
            "module_transition_alpha": dict(self.last_transition_alpha),
            "transition_progress": float(self.last_transition_progress),
            "pending_update": deepcopy(self.pending_update),
            "last_update_events": deepcopy(self.last_update_events),
            "last_trigger_decision": deepcopy(self.last_trigger_decision),
            "feedback_window_history": deepcopy(self.feedback_window_history),
            "module_window_history": deepcopy(self.module_window_history),
        }

    def get_active_modules(self) -> List[Dict]:
        return [spec for spec in self.module_specs if spec["status"] in ("active", "candidate")]

    def _build_features(self, actions: th.Tensor, avail_actions: th.Tensor) -> Dict[str, th.Tensor]:
        action_ids = actions.squeeze(-1).long()
        attack_avail = (avail_actions[..., self.attack_action_start:].sum(dim=-1) > 0).float()
        chose_attack = (action_ids >= self.attack_action_start).float()
        chose_move = ((action_ids >= 2) & (action_ids <= 5)).float()
        chose_idle = (action_ids <= 1).float()

        attack_dim = max(avail_actions.size(-1) - self.attack_action_start, 1)
        attack_indices = (action_ids - self.attack_action_start).clamp(min=0, max=attack_dim - 1)
        attack_onehot = F.one_hot(attack_indices, num_classes=attack_dim).float() * chose_attack.unsqueeze(-1)
        target_counts = attack_onehot.sum(dim=2)
        target_counts_expanded = target_counts.unsqueeze(2).expand(-1, -1, action_ids.size(2), -1)
        same_target_count = th.gather(target_counts_expanded, dim=-1, index=attack_indices.unsqueeze(-1)).squeeze(-1)
        has_teammate_same_target = (same_target_count >= 2).float() * chose_attack

        prev_action_ids = th.zeros_like(action_ids)
        if action_ids.size(1) > 1:
            prev_action_ids[:, 1:] = action_ids[:, :-1]
        prev_chose_attack = (prev_action_ids >= self.attack_action_start).float()
        same_target_as_prev = (action_ids == prev_action_ids).float() * chose_attack * prev_chose_attack
        switched_target = (action_ids != prev_action_ids).float() * chose_attack * prev_chose_attack
        same_action_as_prev = (action_ids == prev_action_ids).float()
        team_attack_ratio = chose_attack.mean(dim=2, keepdim=True).expand_as(chose_attack)
        same_target_ratio = (same_target_count / max(avail_actions.size(2), 1)).float() * chose_attack

        return {
            "attack_avail": attack_avail,
            "chose_attack": chose_attack,
            "chose_move": chose_move,
            "chose_idle": chose_idle,
            "has_teammate_same_target": has_teammate_same_target,
            "same_target_as_prev": same_target_as_prev,
            "switched_target": switched_target,
            "same_action_as_prev": same_action_as_prev,
            "team_attack_ratio": team_attack_ratio,
            "same_target_ratio": same_target_ratio,
        }

    def _safe_torch_namespace(self):
        namespace = {name: getattr(th, name) for name in ALLOWED_TORCH_FUNCTIONS if hasattr(th, name)}
        namespace["tensor"] = th.tensor
        return SimpleNamespace(**namespace)

    def _compile_module_spec(self, spec: Dict) -> Tuple[Optional[callable], Dict]:
        source = self._sanitize_module_source(str(spec.get("python_function_source", "")).strip())
        spec["python_function_source"] = source
        if not source:
            return None, {"ok": False, "reason": "empty_source"}
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return None, {"ok": False, "reason": "syntax_error", "detail": str(exc)}

        validation_error = self._validate_ast(tree)
        if validation_error is not None:
            return None, {"ok": False, "reason": validation_error}

        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1 or functions[0].name != "compute_reward_module":
            return None, {"ok": False, "reason": "expected_single_compute_reward_module"}
        if len(functions[0].args.args) != 2:
            return None, {"ok": False, "reason": "invalid_signature"}

        namespace = {
            "__builtins__": SAFE_BUILTINS,
            "th": self._safe_torch_namespace(),
            "torch": self._safe_torch_namespace(),
            "math": math,
        }
        local_namespace = {}
        try:
            exec(compile(tree, filename="<modular_reward_module>", mode="exec"), namespace, local_namespace)
        except Exception as exc:
            return None, {"ok": False, "reason": "exec_failed", "detail": str(exc)}
        fn = local_namespace.get("compute_reward_module") or namespace.get("compute_reward_module")
        if fn is None:
            return None, {"ok": False, "reason": "function_not_defined"}
        return fn, {"ok": True, "reason": "validated"}

    def _sanitize_module_source(self, source: str) -> str:
        if not source:
            return source
        lines = []
        for line in source.splitlines():
            stripped = line.strip()
            if stripped in {"import torch", "import torch as th"}:
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def _validate_ast(self, tree: ast.AST) -> Optional[str]:
        for node in ast.walk(tree):
            if isinstance(node, FORBIDDEN_AST_NODES):
                return "forbidden_syntax:{}".format(type(node).__name__)
            if isinstance(node, ast.Subscript):
                key_check = self._validate_key_access(node)
                if key_check is not None:
                    return key_check
            if isinstance(node, ast.Call):
                fn_name = self._resolve_call_name(node.func)
                if fn_name in FORBIDDEN_CALL_NAMES:
                    return "forbidden_call:{}".format(fn_name)
                if isinstance(node.func, ast.Attribute):
                    root_name = self._resolve_root_name(node.func)
                    attr_name = node.func.attr
                    if attr_name.startswith("__"):
                        return "forbidden_dunder_attribute:{}".format(attr_name)
                    if root_name in FORBIDDEN_ATTRIBUTE_ROOTS:
                        return "forbidden_attribute_root:{}".format(root_name)
                    if root_name in {"torch", "th"} and attr_name not in ALLOWED_TORCH_FUNCTIONS:
                        return "forbidden_torch_call:{}".format(attr_name)
            if isinstance(node, ast.Attribute):
                root_name = self._resolve_root_name(node)
                if node.attr.startswith("__"):
                    return "forbidden_dunder_attribute:{}".format(node.attr)
                if root_name in FORBIDDEN_ATTRIBUTE_ROOTS:
                    return "forbidden_attribute_root:{}".format(root_name)
        return None

    def _validate_key_access(self, node: ast.Subscript) -> Optional[str]:
        if not isinstance(node.value, ast.Name):
            return None
        root = node.value.id
        key = self._extract_subscript_key(node)
        if root == "features":
            if key is None:
                return "dynamic_feature_key_not_allowed"
            if key not in ALLOWED_FEATURE_KEYS:
                return "unknown_feature_key:{}".format(key)
        elif root == "tensors":
            if key is None:
                return "dynamic_tensor_key_not_allowed"
            if key not in ALLOWED_TENSOR_KEYS:
                return "unknown_tensor_key:{}".format(key)
        return None

    def _extract_subscript_key(self, node: ast.Subscript) -> Optional[str]:
        slice_node = node.slice
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            return slice_node.value
        if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):
            value = slice_node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
        return None

    def _resolve_call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._resolve_root_name(node)
            return "{}.{}".format(base, node.attr) if base else node.attr
        return ""

    def _resolve_root_name(self, node: ast.AST) -> str:
        current = node
        while isinstance(current, ast.Attribute):
            current = current.value
        if isinstance(current, ast.Name):
            return current.id
        return ""

    def _execute_module_function(self, spec: Dict, fn, features: Dict[str, th.Tensor], tensors: Dict[str, th.Tensor]) -> th.Tensor:
        output = fn(features, tensors)
        if not isinstance(output, th.Tensor):
            raise TypeError("module output must be torch.Tensor")
        output = output.float()
        output = th.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        output = output.clamp(
            min=-float(_get_arg(self.args, "modular_reward_module_output_clip", "tactic_module_output_clip", 2.0)),
            max=float(_get_arg(self.args, "modular_reward_module_output_clip", "tactic_module_output_clip", 2.0)),
        )
        expected_shape = features["attack_avail"].shape
        if output.shape != expected_shape:
            raise ValueError("module output shape {} != {}".format(tuple(output.shape), tuple(expected_shape)))
        if spec["status"] in ("frozen", "deprecated"):
            return th.zeros_like(output)
        transition_alpha = float(spec.get("transition_alpha", 1.0))
        if spec["status"] == "candidate":
            output = output * float(spec.get("candidate_shadow_scale", 0.25))
        output = output * max(0.0, min(transition_alpha, 1.0))
        return output * float(spec["scale"])

    def _build_execution_tensors(self, actions: th.Tensor, avail_actions: th.Tensor) -> Dict[str, th.Tensor]:
        return {
            "actions": actions,
            "avail_actions": avail_actions,
            "attack_action_start": self.attack_action_start,
        }

    def compute_module_scores(self, actions: th.Tensor, avail_actions: th.Tensor) -> th.Tensor:
        features = self._build_features(actions, avail_actions)
        tensors = self._build_execution_tensors(actions, avail_actions)
        scores = []
        for spec in self.module_specs:
            fn = self.compiled_module_functions.get(spec["id"])
            if fn is None:
                module_tensor = th.zeros_like(features["attack_avail"])
            else:
                try:
                    module_tensor = self._execute_module_function(spec, fn, features, tensors)
                except Exception as exc:
                    module_tensor = th.zeros_like(features["attack_avail"])
                    spec["validator_report"] = {"ok": False, "reason": "runtime_error", "detail": str(exc)}
            scores.append(module_tensor)
            self.last_module_scores[spec["id"]] = float(module_tensor.mean().item())
        return th.stack(scores, dim=-1)

    def build_selector_context(self, actions: th.Tensor, avail_actions: th.Tensor) -> th.Tensor:
        features = self._build_features(actions, avail_actions)
        context_features = [
            features["attack_avail"],
            features["chose_attack"],
            features["chose_move"],
            features["chose_idle"],
            features["same_target_ratio"],
            features["same_action_as_prev"],
            features["team_attack_ratio"],
        ]
        return th.stack(context_features, dim=-1)

    def record_window_metrics(self, t_env: int, feedback_payload: Optional[Dict] = None):
        max_windows = int(getattr(self.args, "modular_reward_low_signal_windows", 3)) * 4
        for spec in self.module_specs:
            module_id = spec["id"]
            window = {
                "t_env": int(t_env),
                "usage": float(self.last_module_usage.get(module_id, 0.0)),
                "contribution": float(self.last_module_contributions.get(module_id, 0.0)),
                "activation": float(self.last_module_activation.get(module_id, 0.0)),
                "weighted_score": float(self.last_module_weighted_scores.get(module_id, 0.0)),
                "status": spec.get("status", "active"),
            }
            history = self.module_window_history.setdefault(module_id, [])
            history.append(window)
            if len(history) > max_windows:
                del history[:-max_windows]

        if feedback_payload is not None:
            feedback_window = {
                "t_env": int(t_env),
                "latest_test_return": float(feedback_payload.get("latest_test_return", 0.0)),
                "latest_test_win_rate": float(feedback_payload.get("latest_test_win_rate", 0.0)),
                "latest_train_return": float(feedback_payload.get("latest_train_return", 0.0)),
                "selector_entropy": float(feedback_payload.get("selector_entropy", 0.0)),
                "selector_top1_switch_rate": float(feedback_payload.get("selector_top1_switch_rate", 0.0)),
                "selector_balance": float(feedback_payload.get("selector_balance", 0.0)),
            }
            self.feedback_window_history.append(feedback_window)
            if len(self.feedback_window_history) > max_windows:
                del self.feedback_window_history[:-max_windows]

    def _is_training_stagnant(self) -> bool:
        required_windows = int(getattr(self.args, "modular_reward_low_signal_windows", 3))
        min_return_delta = float(getattr(self.args, "modular_reward_stagnation_min_return_delta", 0.02))
        min_win_rate_delta = float(getattr(self.args, "modular_reward_stagnation_min_win_rate_delta", 0.01))
        max_entropy_delta = float(getattr(self.args, "modular_reward_stagnation_max_entropy_delta", 0.02))
        history = self.feedback_window_history
        if len(history) < required_windows:
            return False

        recent = history[-required_windows:]
        return_span = max(item["latest_test_return"] for item in recent) - min(item["latest_test_return"] for item in recent)
        win_rate_span = max(item["latest_test_win_rate"] for item in recent) - min(item["latest_test_win_rate"] for item in recent)
        entropy_span = max(item["selector_entropy"] for item in recent) - min(item["selector_entropy"] for item in recent)
        return return_span <= min_return_delta and win_rate_span <= min_win_rate_delta and entropy_span <= max_entropy_delta

    def _selector_collapsing(self) -> bool:
        required_windows = int(getattr(self.args, "modular_reward_low_signal_windows", 3))
        entropy_threshold = float(getattr(self.args, "modular_reward_collapse_entropy_threshold", 0.9))
        switch_threshold = float(getattr(self.args, "modular_reward_collapse_switch_threshold", 0.08))
        history = self.feedback_window_history
        if len(history) < required_windows:
            return False

        recent = history[-required_windows:]
        mean_entropy = sum(item["selector_entropy"] for item in recent) / float(required_windows)
        mean_switch = sum(item["selector_top1_switch_rate"] for item in recent) / float(required_windows)
        return mean_entropy <= entropy_threshold and mean_switch <= switch_threshold

    def _is_training_improving(self) -> bool:
        required_windows = int(getattr(self.args, "modular_reward_low_signal_windows", 3))
        min_return_gain = float(getattr(self.args, "modular_reward_improving_min_return_gain", 0.05))
        min_win_rate_gain = float(getattr(self.args, "modular_reward_improving_min_win_rate_gain", 0.01))
        history = self.feedback_window_history
        if len(history) < required_windows:
            return False

        recent = history[-required_windows:]
        return_gain = float(recent[-1].get("latest_test_return", 0.0)) - float(recent[0].get("latest_test_return", 0.0))
        win_rate_gain = float(recent[-1].get("latest_test_win_rate", 0.0)) - float(recent[0].get("latest_test_win_rate", 0.0))
        return return_gain >= min_return_gain or win_rate_gain >= min_win_rate_gain

    def _selector_structurally_biased(self) -> bool:
        required_windows = int(getattr(self.args, "modular_reward_low_signal_windows", 3))
        balance_threshold = float(getattr(self.args, "modular_reward_structural_bias_balance_threshold", 0.02))
        entropy_threshold = float(getattr(self.args, "modular_reward_structural_bias_entropy_threshold", 1.1))
        usage_gap_threshold = float(getattr(self.args, "modular_reward_structural_bias_usage_gap_threshold", 0.35))
        history = self.feedback_window_history
        if len(history) < required_windows:
            return False

        recent = history[-required_windows:]
        mean_balance = sum(float(item.get("selector_balance", 0.0)) for item in recent) / float(required_windows)
        mean_entropy = sum(float(item.get("selector_entropy", 0.0)) for item in recent) / float(required_windows)

        active_specs = []
        for spec in self.module_specs:
            if spec.get("status") != "active":
                continue
            module_history = self.module_window_history.get(spec["id"], [])
            if len(module_history) < required_windows:
                continue
            active_specs.append(module_history[-required_windows:])

        usage_gap = 0.0
        if len(active_specs) >= 2:
            mean_usages = [sum(float(item.get("usage", 0.0)) for item in module_recent) / float(required_windows) for module_recent in active_specs]
            usage_gap = max(mean_usages) - min(mean_usages)

        return mean_balance >= balance_threshold or (mean_entropy <= entropy_threshold and usage_gap >= usage_gap_threshold)

    def _candidate_severity(self, spec: Dict, recent: List[Dict], usage_threshold: float, contribution_threshold: float, activation_threshold: float) -> float:
        return sum(
            (usage_threshold - float(item.get("usage", 0.0)))
            + (contribution_threshold - float(item.get("contribution", 0.0)))
            + (activation_threshold - float(item.get("activation", 0.0)))
            for item in recent
        )

    def _relative_low_signal_target(self, t_env: int) -> Optional[Dict]:
        required_windows = int(getattr(self.args, "modular_reward_low_signal_windows", 3))
        min_usage = float(getattr(self.args, "modular_reward_relative_usage_ceiling", 0.28))
        min_activation = float(getattr(self.args, "modular_reward_relative_activation_ceiling", 0.25))
        contribution_margin = float(getattr(self.args, "modular_reward_relative_contribution_margin", 0.01))
        candidates = []

        active_specs = []
        for spec in self.module_specs:
            if spec.get("status") != "active":
                continue
            if t_env < int(spec.get("cooldown_until", 0)):
                continue
            history = self.module_window_history.get(spec["id"], [])
            if len(history) < required_windows:
                continue
            active_specs.append((spec, history[-required_windows:]))

        if len(active_specs) < 2:
            return None

        mean_contributions = [sum(item.get("contribution", 0.0) for item in recent) / float(required_windows) for _, recent in active_specs]
        contribution_floor = min(mean_contributions) + contribution_margin

        for spec, recent in active_specs:
            mean_usage = sum(item.get("usage", 0.0) for item in recent) / float(required_windows)
            mean_activation = sum(item.get("activation", 0.0) for item in recent) / float(required_windows)
            mean_contribution = sum(item.get("contribution", 0.0) for item in recent) / float(required_windows)
            if mean_usage <= min_usage and mean_activation <= min_activation and mean_contribution <= contribution_floor:
                severity = (
                    (min_usage - mean_usage)
                    + (min_activation - mean_activation)
                    + max(contribution_floor - mean_contribution, 0.0)
                )
                candidates.append((severity, spec))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _choose_update_target(self, t_env: int) -> Tuple[Optional[Dict], Dict]:
        absolute_target = self._low_signal_target(t_env)
        if absolute_target is not None:
            return absolute_target, {
                "mode": "absolute_low_signal",
                "t_env": int(t_env),
                "stagnant": False,
                "collapse": False,
                "improving": False,
                "structural_bias": False,
            }

        stagnant = self._is_training_stagnant()
        collapse = self._selector_collapsing()
        improving = self._is_training_improving()
        structural_bias = self._selector_structurally_biased()
        if stagnant or collapse:
            relative_target = self._relative_low_signal_target(t_env)
            if relative_target is not None:
                return relative_target, {
                    "mode": "relative_low_signal",
                    "t_env": int(t_env),
                    "stagnant": bool(stagnant),
                    "collapse": bool(collapse),
                    "improving": bool(improving),
                    "structural_bias": bool(structural_bias),
                }

        if improving and structural_bias:
            relative_target = self._relative_low_signal_target(t_env)
            if relative_target is not None:
                return relative_target, {
                    "mode": "relative_low_signal_improving",
                    "t_env": int(t_env),
                    "stagnant": bool(stagnant),
                    "collapse": bool(collapse),
                    "improving": bool(improving),
                    "structural_bias": bool(structural_bias),
                }

        return None, {
            "mode": "no_trigger",
            "t_env": int(t_env),
            "stagnant": bool(stagnant),
            "collapse": bool(collapse),
            "improving": bool(improving),
            "structural_bias": bool(structural_bias),
        }

    def _low_signal_target(self, t_env: int) -> Optional[Dict]:
        required_windows = int(getattr(self.args, "modular_reward_low_signal_windows", 3))
        usage_threshold = float(getattr(self.args, "modular_reward_low_usage_threshold", 0.12))
        contribution_threshold = float(getattr(self.args, "modular_reward_low_contribution_threshold", 0.0))
        activation_threshold = float(getattr(self.args, "modular_reward_low_activation_threshold", 0.1))

        candidates = []
        for spec in self.module_specs:
            if spec.get("status") != "active":
                continue
            if t_env < int(spec.get("cooldown_until", 0)):
                continue
            history = self.module_window_history.get(spec["id"], [])
            if len(history) < required_windows:
                continue
            recent = history[-required_windows:]
            if all(
                float(item.get("usage", 0.0)) <= usage_threshold
                and float(item.get("contribution", 0.0)) <= contribution_threshold
                and float(item.get("activation", 0.0)) <= activation_threshold
                for item in recent
            ):
                severity = self._candidate_severity(
                    spec,
                    recent,
                    usage_threshold,
                    contribution_threshold,
                    activation_threshold,
                )
                candidates.append((severity, spec))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _make_candidate_spec(self, proposal: Dict, target_spec: Dict, t_env: int, validation_report: Optional[Dict]) -> Dict:
        version = max(int(target_spec.get("version", 1)) + 1, int(proposal.get("version_parent", 1)) + 1)
        return self._normalise_spec({
            "id": target_spec["id"],
            "name": proposal["name"],
            "version": version,
            "status": "candidate",
            "target_status": "active",
            "scale": proposal.get("scale", target_spec.get("scale", 1.0)),
            "description": proposal.get("description", ""),
            "when_to_use": proposal.get("when_to_use", ""),
            "created_at_t_env": int(target_spec.get("created_at_t_env", t_env)),
            "last_updated_t_env": int(t_env),
            "promotion_score": float(target_spec.get("promotion_score", 0.0)),
            "rollback_count": int(target_spec.get("rollback_count", 0)),
            "cooldown_until": int(t_env + int(getattr(self.args, "modular_reward_module_cooldown", 200000))),
            "source_type": "generated",
            "parent_id": target_spec["id"],
            "fallback_id": target_spec.get("fallback_id", target_spec["id"]),
            "required_inputs": proposal.get("required_inputs", []),
            "hypothesis": proposal.get("hypothesis", ""),
            "expected_effect": proposal.get("expected_effect", ""),
            "python_function_source": proposal.get("python_function_source", ""),
            "validator_report": validation_report,
            "candidate_shadow_scale": float(getattr(self.args, "modular_reward_candidate_shadow_scale", 0.25)),
        }, index=0)

    def _validate_candidate_with_dry_run(self, proposal: Dict, target_spec: Dict, t_env: int) -> Tuple[Dict, Dict]:
        candidate_spec = self._make_candidate_spec(proposal, target_spec, t_env, validation_report=None)
        orthogonality_report = self._orthogonality_report(candidate_spec, target_spec)
        if not orthogonality_report.get("ok", False):
            validation_report = {
                "static_validation": {"ok": True, "reason": "validated"},
                "orthogonality": orthogonality_report,
                "dry_run": {"ok": False, "reason": "orthogonality_rejected"},
            }
            candidate_spec["validator_report"] = validation_report
            return candidate_spec, validation_report
        required_inputs = set(candidate_spec.get("required_inputs", []))
        allowed_inputs = ALLOWED_FEATURE_KEYS | ALLOWED_TENSOR_KEYS
        unknown_inputs = sorted(required_inputs - allowed_inputs)
        if unknown_inputs:
            validation_report = {
                "static_validation": {"ok": False, "reason": "unknown_required_inputs", "detail": unknown_inputs},
                "orthogonality": orthogonality_report,
                "dry_run": {"ok": False, "reason": "static_validation_failed"},
            }
            candidate_spec["validator_report"] = validation_report
            return candidate_spec, validation_report

        fn, report = self._compile_module_spec(candidate_spec)
        if not report.get("ok", False):
            validation_report = {
                "static_validation": report,
                "orthogonality": orthogonality_report,
                "dry_run": {"ok": False, "reason": "static_validation_failed"},
            }
            candidate_spec["validator_report"] = validation_report
            return candidate_spec, validation_report

        dry_run = self._runtime_dry_run_report(candidate_spec, fn)
        validation_report = {
            "static_validation": report,
            "orthogonality": orthogonality_report,
            "dry_run": dry_run,
        }
        candidate_spec["validator_report"] = validation_report
        return candidate_spec, validation_report

    def _orthogonality_report(self, candidate_spec: Dict, target_spec: Dict) -> Dict:
        target_family = self.initializer._module_family_label(target_spec)
        candidate_family = self.initializer._module_family_label(candidate_spec)
        same_family_others = [
            spec for spec in self.module_specs
            if spec.get("id") != target_spec.get("id") and self.initializer._module_family_label(spec) == candidate_family
        ]
        allow_same_family = bool(getattr(self.args, "modular_reward_allow_same_family_replacement", False))
        if not allow_same_family and candidate_family == target_family and same_family_others:
            return {
                "ok": False,
                "reason": "redundant_family_replacement",
                "candidate_family": candidate_family,
                "target_family": target_family,
                "same_family_count": len(same_family_others),
            }
        if not allow_same_family and candidate_family != "other" and len(same_family_others) >= int(getattr(self.args, "modular_reward_max_modules_per_family", 2)):
            return {
                "ok": False,
                "reason": "family_capacity_reached",
                "candidate_family": candidate_family,
                "target_family": target_family,
                "same_family_count": len(same_family_others),
            }
        return {
            "ok": True,
            "reason": "orthogonal_enough",
            "candidate_family": candidate_family,
            "target_family": target_family,
            "same_family_count": len(same_family_others),
        }

    def maybe_advance_transition(self, t_env: int) -> Optional[Dict]:
        if self.pending_update is None:
            self.last_transition_progress = 1.0
            return None
        transition_steps = max(int(getattr(self.args, "modular_reward_transition_steps", 50000)), 1)
        start_t = int(self.pending_update.get("transition_start_t_env", self.pending_update.get("t_env", t_env)))
        progress = min(max((t_env - start_t) / float(transition_steps), 0.0), 1.0)
        candidate_id = self.pending_update.get("candidate_id")
        for spec in self.module_specs:
            if spec["id"] == candidate_id:
                spec["transition_alpha"] = progress
                break
        self.last_transition_progress = progress
        self._sync_runtime_metadata()
        return {
            "candidate_id": candidate_id,
            "transition_progress": progress,
        }

    def apply_feedback_update(self, feedback_payload: Dict, t_env: int) -> List[Dict]:
        events = []
        enable_lifecycle = bool(getattr(self.args, "modular_reward_enable_lifecycle", True))
        if not enable_lifecycle:
            self.last_update_events = []
            return events

        latest_return = float(feedback_payload.get("latest_test_return", 0.0))
        latest_win_rate = float(feedback_payload.get("latest_test_win_rate", 0.0))
        rollback_threshold = float(getattr(self.args, "modular_reward_rollback_threshold", -0.02))
        promote_threshold = float(getattr(self.args, "modular_reward_promote_threshold", 0.0))
        cooldown = int(getattr(self.args, "modular_reward_module_cooldown", 200000))

        if self.pending_update is not None:
            return_delta = latest_return - float(self.pending_update.get("baseline_return", 0.0))
            win_rate_delta = latest_win_rate - float(self.pending_update.get("baseline_win_rate", 0.0))
            candidate_id = self.pending_update.get("candidate_id")
            parent_id = self.pending_update.get("parent_id")
            if return_delta < rollback_threshold or win_rate_delta < rollback_threshold:
                self.module_specs = [deepcopy(spec) for spec in self.pending_update["snapshot"]]
                for spec in self.module_specs:
                    if spec["id"] == parent_id:
                        spec["status"] = "active"
                        spec["cooldown_until"] = int(t_env + cooldown)
                        spec["rollback_count"] = int(spec.get("rollback_count", 0)) + 1
                self.pending_update = None
                self._refresh_module_cache()
                event = {
                    "type": "rollback",
                    "t_env": int(t_env),
                    "candidate_id": candidate_id,
                    "parent_id": parent_id,
                    "return_delta": float(return_delta),
                    "win_rate_delta": float(win_rate_delta),
                }
                events.append(event)
                self.last_update_events = events
                return events

            if return_delta >= promote_threshold or win_rate_delta >= promote_threshold:
                for spec in self.module_specs:
                    if spec["id"] == candidate_id:
                        spec["status"] = "active"
                        spec["target_status"] = "active"
                        spec["transition_alpha"] = 1.0
                        spec["promotion_score"] = float(spec.get("promotion_score", 0.0)) + max(return_delta, 0.0)
                        spec["last_updated_t_env"] = int(t_env)
                        spec["cooldown_until"] = int(t_env + cooldown)
                self.pending_update = None
                self._refresh_module_cache()
                event = {
                    "type": "promote",
                    "t_env": int(t_env),
                    "candidate_id": candidate_id,
                    "parent_id": parent_id,
                    "return_delta": float(return_delta),
                    "win_rate_delta": float(win_rate_delta),
                }
                events.append(event)
                self.last_update_events = events
                return events

            event = {
                "type": "candidate_hold",
                "t_env": int(t_env),
                "candidate_id": candidate_id,
                "parent_id": parent_id,
                "return_delta": float(return_delta),
                "win_rate_delta": float(win_rate_delta),
            }
            events.append(event)
            self.last_update_events = events
            return events

        target_spec, trigger_decision = self._choose_update_target(t_env)
        self.last_trigger_decision = deepcopy(trigger_decision)
        if target_spec is None:
            self.last_update_events = []
            return events

        interface_payload = self.get_interface_payload()
        current_specs = self.get_specs_payload()
        proposal_result = self.initializer.propose_replacement(
            current_specs=current_specs,
            stats_payload=feedback_payload,
            env_name=getattr(self.args, "env", "unknown"),
            env_args=getattr(self.args, "env_args", {}),
            interface_payload=interface_payload,
            target_module=deepcopy(target_spec),
        )
        if proposal_result.get("action") != "replace_module" or proposal_result.get("proposal") is None:
            target_spec["cooldown_until"] = int(t_env + cooldown)
            event = {
                "type": "llm_keep",
                "t_env": int(t_env),
                "target_id": target_spec["id"],
                "llm_source": proposal_result.get("source", "default"),
                "trigger_decision": deepcopy(self.last_trigger_decision),
            }
            events.append(event)
            self.last_update_events = events
            return events

        candidate_spec, validation_report = self._validate_candidate_with_dry_run(proposal_result["proposal"], target_spec, t_env)
        if not validation_report.get("static_validation", {}).get("ok", False) or not validation_report.get("dry_run", {}).get("ok", False):
            target_spec["cooldown_until"] = int(t_env + cooldown)
            event = {
                "type": "reject",
                "t_env": int(t_env),
                "target_id": target_spec["id"],
                "candidate_name": candidate_spec["name"],
                "validation": validation_report,
                "candidate_source": candidate_spec.get("python_function_source", ""),
                "llm_source": proposal_result.get("source", "default"),
                "trigger_decision": deepcopy(self.last_trigger_decision),
            }
            events.append(event)
            self.last_update_events = events
            return events

        candidate_spec["transition_alpha"] = 0.0
        snapshot = deepcopy(self.module_specs)
        for index, spec in enumerate(self.module_specs):
            if spec["id"] == target_spec["id"]:
                self.module_specs[index] = candidate_spec
                break
        self.module_window_history.setdefault(candidate_spec["id"], [])
        self.pending_update = {
            "snapshot": snapshot,
            "baseline_return": latest_return,
            "baseline_win_rate": latest_win_rate,
            "t_env": int(t_env),
            "transition_start_t_env": int(t_env),
            "candidate_id": candidate_spec["id"],
            "parent_id": target_spec["id"],
            "target_name": target_spec["name"],
            "candidate_name": candidate_spec["name"],
            "candidate_source": candidate_spec.get("python_function_source", ""),
            "candidate_spec": deepcopy(candidate_spec),
            "validation": validation_report,
            "llm_source": proposal_result.get("source", "default"),
            "llm_prompt": proposal_result.get("prompt", ""),
            "llm_raw_response": proposal_result.get("raw_response", ""),
        }
        self._refresh_module_cache()
        event = {
            "type": "stage_candidate",
            "t_env": int(t_env),
            "target_id": target_spec["id"],
            "candidate_id": candidate_spec["id"],
            "candidate_name": candidate_spec["name"],
            "validation": validation_report,
            "llm_source": proposal_result.get("source", "default"),
            "trigger_decision": deepcopy(self.last_trigger_decision),
        }
        events.append(event)
        self.last_update_events = events
        return events

    def _sync_runtime_metadata(self):
        self.last_module_status = {spec["id"]: spec["status"] for spec in self.module_specs}
        self.last_module_status_code = {spec["id"]: STATUS_TO_CODE.get(spec["status"], 0.0) for spec in self.module_specs}
        self.last_module_scale = {spec["id"]: float(spec["scale"]) for spec in self.module_specs}
        self.last_transition_alpha = {spec["id"]: float(spec.get("transition_alpha", 1.0)) for spec in self.module_specs}
