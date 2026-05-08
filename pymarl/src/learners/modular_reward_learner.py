import copy
import json
import math
import os

import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop

from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
from modules.modular_reward_ivf import ContextValueNetwork, IVFTrainer, OuterLoopRewardSelector, SelectorNetwork, TrainingStateTracker
from modules.modular_reward_pool import ModularRewardModulePool


class ModularRewardQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

        self.selector_hidden_dim = getattr(args, "modular_reward_selector_hidden_dim", 64)
        self.selector_balance_coef = getattr(args, "modular_reward_selector_balance_coef", 0.01)
        self.selector_activation_threshold = getattr(args, "modular_reward_selector_activation_threshold", 0.35)
        self.selector_feedback_coef = float(getattr(args, "modular_reward_selector_feedback_coef", 0.5))
        self.selector_feedback_value_coef = float(getattr(args, "modular_reward_selector_feedback_value_coef", 0.5))
        self.selector_feedback_target_temperature = float(getattr(args, "modular_reward_selector_feedback_target_temperature", 2.0))
        self.selector_feedback_trust_prior_coef = float(getattr(args, "modular_reward_selector_feedback_trust_prior_coef", 0.35))
        self.selector_trust_max_gate = float(getattr(args, "modular_reward_selector_trust_max_gate", 0.75))

        self.modular_reward_beta = getattr(args, "modular_reward_beta", getattr(args, "tactic_reward_scale", 0.3))
        self.modular_reward_beta_start = float(getattr(args, "modular_reward_beta_start", self.modular_reward_beta))
        self.modular_reward_beta_end = float(getattr(args, "modular_reward_beta_end", self.modular_reward_beta))
        self.modular_reward_beta_ramp_start_steps = getattr(args, "modular_reward_beta_ramp_start_steps", getattr(args, "modular_reward_guidance_start_steps", 0))
        self.modular_reward_beta_ramp_steps = getattr(args, "modular_reward_beta_ramp_steps", getattr(args, "modular_reward_guidance_ramp_steps", 0))
        self.selector_temperature = getattr(args, "modular_reward_selector_temperature", 1.0)
        self.selector_entropy_coef = getattr(args, "modular_reward_selector_entropy_coef", 0.01)
        self.selector_smoothing_coef = getattr(args, "modular_reward_selector_smoothing_coef", 0.02)
        self.modular_reward_shaping_delta_clip = float(getattr(args, "modular_reward_shaping_delta_clip", 0.5))
        self.modular_reward_check_interval = getattr(args, "modular_reward_check_interval", getattr(args, "tactic_check_interval", 300))
        self.modular_reward_warmup_steps = getattr(args, "modular_reward_warmup_steps", getattr(args, "tactic_warmup_steps", 0))
        self.modular_reward_module_update_interval = getattr(args, "modular_reward_module_update_interval", getattr(args, "modular_reward_refinement_interval", 0))
        self.modular_reward_enable_ivf = getattr(args, "modular_reward_enable_ivf", False)
        self.modular_reward_guidance_start_steps = getattr(args, "modular_reward_guidance_start_steps", 0)
        self.modular_reward_guidance_ramp_steps = getattr(args, "modular_reward_guidance_ramp_steps", 0)
        self.modular_reward_min_beta_ratio = min(
            max(float(getattr(args, "modular_reward_min_beta_ratio", 1.0 if self.modular_reward_guidance_ramp_steps <= 0 else 0.0)), 0.0),
            1.0,
        )
        self.training_iters = getattr(args, "training_iters", 1)

        self.outer_selector_enabled = getattr(args, "modular_reward_use_outer_loop_selector", False)
        self.outer_selector_mix_coef = getattr(args, "modular_reward_outer_selector_mix_coef", 0.5)
        self.outer_selector_interval = getattr(args, "modular_reward_outer_selector_interval", self.modular_reward_check_interval)
        self.outer_selector_return_scale = getattr(args, "modular_reward_outer_selector_return_scale", 20.0)

        self.module_pool = ModularRewardModulePool(args, logger)
        self.module_pool.save_specs(getattr(args, "unique_token", "modular_reward_init"), args.local_results_path)
        self.selector_net = SelectorNetwork(
            input_dim=self.module_pool.selector_input_dim(),
            n_modules=self.module_pool.num_modules(),
            hidden_dim=self.selector_hidden_dim,
        )
        self.selector_feedback_net = SelectorNetwork(
            input_dim=self.module_pool.selector_input_dim(),
            n_modules=self.module_pool.num_modules(),
            hidden_dim=self.selector_hidden_dim,
        )
        self.selector_value_net = ContextValueNetwork(
            input_dim=self.module_pool.selector_input_dim(),
            hidden_dim=self.selector_hidden_dim,
        )
        self.selector_feedback_value_net = ContextValueNetwork(
            input_dim=self.module_pool.selector_input_dim(),
            hidden_dim=self.selector_hidden_dim,
        )
        self.selector_trust_gate_net = ContextValueNetwork(
            input_dim=self.module_pool.selector_input_dim(),
            hidden_dim=self.selector_hidden_dim,
        )
        self.selector_params = (
            list(self.selector_net.parameters())
            + list(self.selector_feedback_net.parameters())
            + list(self.selector_value_net.parameters())
            + list(self.selector_feedback_value_net.parameters())
            + list(self.selector_trust_gate_net.parameters())
        )
        self.selector_optimizer = RMSprop(
            params=self.selector_params,
            lr=getattr(args, "modular_reward_selector_lr", args.lr),
            alpha=args.optim_alpha,
            eps=args.optim_eps,
        )

        self.state_tracker = TrainingStateTracker(window=200)
        self.ivf_trainer = IVFTrainer(
            n_actions=3,
            lr=getattr(args, "modular_reward_ivf_lr", 1e-3),
            gamma=getattr(args, "modular_reward_ivf_gamma", 0.5),
            epsilon_start=getattr(args, "modular_reward_ivf_epsilon_start", getattr(args, "ivf_epsilon_start", 0.5)),
            epsilon_end=getattr(args, "modular_reward_ivf_epsilon_end", getattr(args, "ivf_epsilon_end", 0.1)),
            epsilon_decay_steps=getattr(args, "modular_reward_ivf_epsilon_decay", getattr(args, "ivf_epsilon_decay", 80)),
        )
        self.outer_loop_selector = OuterLoopRewardSelector(
            state_dim=8,
            n_modules=self.module_pool.num_modules(),
            hidden_dim=getattr(args, "modular_reward_outer_selector_hidden_dim", self.selector_hidden_dim),
            lr=getattr(args, "modular_reward_outer_selector_lr", getattr(args, "modular_reward_selector_lr", args.lr)),
            entropy_coef=getattr(args, "modular_reward_outer_selector_entropy_coef", 0.01),
            value_coef=getattr(args, "modular_reward_outer_selector_value_coef", 0.5),
            min_alpha=getattr(args, "modular_reward_outer_selector_min_alpha", 0.3),
            return_scale=self.outer_selector_return_scale,
        )

        self.selector_modes = ["env_only", "soft_routing", "explore_bonus"]
        self.selector_mode = 1
        self.prev_selector_weights = None
        self._pending_ivf_state = None
        self._pending_ivf_action = 0
        self._ivf_last_decision_episode = 0
        self._last_eval_return = 0.0
        self._pending_outer_state = None
        self._pending_outer_prior = None
        self._pending_outer_eval_return = 0.0
        self._last_outer_selector_t_env = 0
        self.current_outer_prior = None
        self.current_outer_alpha = None
        self._selector_feedback_context_sum = None
        self._selector_feedback_exposure_sum = None
        self._selector_feedback_batches = 0
        self.latest_selector_entropy = 0.0
        self.latest_aux_reward_mean = 0.0
        self.latest_selector_weight_mean = 0.0
        self.latest_ivf_loss = 0.0
        self.latest_train_return = 0.0
        self.latest_selector_advantage = 0.0
        self.latest_selector_balance = 0.0
        self.latest_selector_value_loss = 0.0
        self.latest_selector_feedback_loss = 0.0
        self.latest_selector_feedback_value_loss = 0.0
        self.latest_selector_feedback_advantage = 0.0
        self.latest_selector_feedback_target = 0.0
        self.latest_selector_top1_switch_rate = 0.0
        self.latest_selector_weight_std = 0.0
        self.latest_selector_time_std = 0.0
        self.latest_selector_agent_std = 0.0
        self.latest_outer_selector_loss = 0.0
        self.latest_outer_selector_policy_loss = 0.0
        self.latest_outer_selector_value_loss = 0.0
        self.latest_outer_selector_entropy = 0.0
        self.latest_outer_selector_advantage = 0.0
        self.latest_outer_selector_return_delta = 0.0
        self.latest_outer_selector_return_delta_norm = 0.0
        self.latest_outer_selector_value = 0.0
        self.latest_outer_selector_alpha_mean = 0.0
        self.latest_outer_selector_alpha_max = 0.0
        self.latest_selector_trace = {}
        self.latest_selector_trust_gate = 0.0
        self.latest_selector_trust_bias_std = 0.0
        self.latest_test_win_rate = 0.0
        self.latest_env_feedback_signal = 0.0
        self.latest_module_update_event_count = 0.0
        self.latest_module_update_trigger_count = 0.0
        self.latest_shaped_reward_delta_mean = 0.0
        self.latest_shaped_reward_delta_clip_ratio = 0.0
        self.latest_transition_loss = 0.0
        self.latest_transition_progress = 1.0
        self.latest_module_update_check_t_env = 0
        self.latest_selector_viz_t_env = 0

    def update_win_stat(self, win_rate: float):
        self.latest_test_win_rate = float(win_rate)
        if self.state_tracker.win_history:
            self.state_tracker.win_history[-1] = float(win_rate)
        else:
            self.state_tracker.win_history.append(float(win_rate))

    def set_test_return(self, test_return: float):
        self._last_eval_return = float(test_return)

    def maybe_ivf_decide(self, episode_num: int, t_env: int):
        if not self.modular_reward_enable_ivf:
            return
        if t_env < self.modular_reward_warmup_steps:
            return
        if episode_num - self._ivf_last_decision_episode < self.modular_reward_check_interval:
            return

        current_state = self.state_tracker.get_state()
        if self._pending_ivf_state is not None:
            delta = self._last_eval_return - self._pending_eval_return
            self.ivf_trainer.store(self._pending_ivf_state, self._pending_ivf_action, delta, current_state)
            self.latest_ivf_loss = self.ivf_trainer.update()

        action = self.ivf_trainer.select_action(current_state)
        self.selector_mode = action
        self.state_tracker.record_action(action)
        self._pending_ivf_state = current_state
        self._pending_ivf_action = action
        self._pending_eval_return = self._last_eval_return
        self._ivf_last_decision_episode = episode_num

    def maybe_update_outer_loop_selector(self, t_env: int):
        if not self.outer_selector_enabled:
            return
        if t_env < self.modular_reward_warmup_steps:
            return
        if t_env - self._last_outer_selector_t_env < self.outer_selector_interval:
            return

        current_state = self.state_tracker.get_state()
        if self._pending_outer_state is not None and self._pending_outer_prior is not None:
            delta = self._last_eval_return - self._pending_outer_eval_return
            stats = self.outer_loop_selector.update(self._pending_outer_state, self._pending_outer_prior, delta)
            self.latest_outer_selector_loss = stats["loss"]
            self.latest_outer_selector_policy_loss = stats["policy_loss"]
            self.latest_outer_selector_value_loss = stats["value_loss"]
            self.latest_outer_selector_entropy = stats["entropy"]
            self.latest_outer_selector_advantage = stats["advantage"]
            self.latest_outer_selector_return_delta = stats["return_delta"]
            self.latest_outer_selector_return_delta_norm = stats["return_delta_norm"]
            self.latest_outer_selector_value = stats["value"]
            self.latest_outer_selector_alpha_mean = stats["alpha_mean"]
            self.latest_outer_selector_alpha_max = stats["alpha_max"]

        sample = self.outer_loop_selector.sample_prior(current_state)
        self.current_outer_prior = sample["prior"]
        self.current_outer_alpha = sample["alpha"]
        self._pending_outer_state = current_state.detach().clone()
        self._pending_outer_prior = sample["prior"].detach().clone()
        self._pending_outer_eval_return = self._last_eval_return
        self._last_outer_selector_t_env = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        guidance_strength = self._guidance_strength(t_env)
        effective_beta = self._effective_beta(t_env)

        selector_context = self.module_pool.build_selector_context(actions, avail_actions[:, :-1])
        module_scores = self.module_pool.compute_module_scores(actions, avail_actions[:, :-1])
        selector_logits = self.selector_net(selector_context) / max(self.selector_temperature, 1e-6)
        selector_feedback_logits = self.selector_feedback_net(selector_context)
        trust_prior = self.module_pool.get_trust_prior_tensor(device=selector_context.device)
        trust_prior = trust_prior.view(1, 1, 1, -1).expand_as(selector_logits)
        trust_bias = th.log(trust_prior.clamp(min=1e-8)) - math.log(1.0 / max(self.module_pool.num_modules(), 1))
        trust_gate = th.sigmoid(self.selector_trust_gate_net(selector_context)).squeeze(-1)
        trust_gate = trust_gate * self.selector_trust_max_gate * guidance_strength
        combined_selector_logits = (
            selector_logits
            + self.selector_feedback_coef * selector_feedback_logits
            + self.selector_feedback_trust_prior_coef * trust_gate.unsqueeze(-1) * trust_bias
        )
        selector_weights = th.softmax(combined_selector_logits, dim=-1)
        selector_values = self.selector_value_net(selector_context).squeeze(-1)

        if self.outer_selector_enabled and self.current_outer_prior is not None:
            prior = self.current_outer_prior.to(selector_weights.device).view(1, 1, 1, -1)
            mix_coef = min(max(float(self.outer_selector_mix_coef), 0.0), 1.0) * guidance_strength
            selector_weights = (1.0 - mix_coef) * selector_weights + mix_coef * prior.expand_as(selector_weights)
            selector_weights = selector_weights / selector_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            combined_selector_logits = th.log(selector_weights.clamp(min=1e-8))

        selector_mode = getattr(self.args, "modular_reward_force_selector_mode", None)
        if selector_mode is None:
            selector_mode = self.selector_mode
        if selector_mode == 0:
            selector_weights = th.zeros_like(selector_weights)
            selector_values = th.zeros_like(selector_values)
            trust_gate = th.zeros_like(trust_gate)
        elif selector_mode == 2:
            selector_weights = th.softmax(combined_selector_logits + 0.2 * guidance_strength, dim=-1)

        module_contributions = selector_weights * module_scores
        transition_payload = self.module_pool.maybe_advance_transition(t_env)
        transition_progress = 1.0 if transition_payload is None else float(transition_payload.get("transition_progress", 1.0))
        transition_target = self.prev_selector_weights if self.prev_selector_weights is not None and self.prev_selector_weights.shape == selector_weights.shape else selector_weights.detach()
        transition_loss = ((selector_weights - transition_target.detach()) ** 2).mean() * (1.0 - transition_progress)
        auxiliary_reward = module_contributions.sum(dim=-1, keepdim=True)
        team_auxiliary_reward = auxiliary_reward.mean(dim=2, keepdim=False)
        shaped_reward_delta = effective_beta * team_auxiliary_reward
        shaped_reward_delta_clip = max(self.modular_reward_shaping_delta_clip, 0.0)
        clipped_shaped_reward_delta = shaped_reward_delta.clamp(min=-shaped_reward_delta_clip, max=shaped_reward_delta_clip)
        shaped_rewards = rewards + clipped_shaped_reward_delta

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            mac_out.append(self.mac.forward(batch, t=t))
        mac_out = th.stack(mac_out, dim=1)

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_mac_out.append(self.target_mac.forward(batch, t=t))
        target_mac_out = th.stack(target_mac_out[1:], dim=1)
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        targets = shaped_rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        td_error = chosen_action_qvals - targets.detach()
        mask_expanded = mask.expand_as(td_error)
        masked_td_error = td_error * mask_expanded
        td_loss = (masked_td_error ** 2).sum() / mask_expanded.sum()

        selector_entropy = -(selector_weights * (selector_weights.clamp(min=1e-8).log())).sum(dim=-1).mean()
        selector_smoothing = th.tensor(0.0, device=td_loss.device)
        if self.prev_selector_weights is not None and self.prev_selector_weights.shape == selector_weights.shape:
            selector_smoothing = ((selector_weights - self.prev_selector_weights) ** 2).mean()

        selector_balance = (selector_weights.mean(dim=(0, 1, 2)) - (1.0 / max(self.module_pool.num_modules(), 1))) ** 2
        selector_balance = selector_balance.mean()
        selector_value_loss = th.tensor(0.0, device=td_loss.device)
        selector_policy_alignment = th.tensor(0.0, device=td_loss.device)

        selector_reg = (
            -0.25 * self.selector_entropy_coef * selector_entropy
            + self.selector_smoothing_coef * selector_smoothing
            + 0.25 * self.selector_balance_coef * selector_balance
            + float(getattr(self.args, "modular_reward_transition_loss_coef", 0.05)) * transition_loss
        )
        selector_loss = selector_reg + selector_value_loss + selector_policy_alignment
        loss = td_loss + selector_loss

        self.optimiser.zero_grad()
        self.selector_optimizer.zero_grad()
        loss.backward()
        base_grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        selector_grad_norm = th.nn.utils.clip_grad_norm_(self.selector_params, self.args.grad_norm_clip)
        self.optimiser.step()
        self.selector_optimizer.step()

        self.prev_selector_weights = selector_weights.detach()
        self._accumulate_selector_feedback_stats(selector_context, module_contributions)
        self._record_module_stats(selector_weights, module_scores, module_contributions)
        self._record_selector_diagnostics(selector_weights)
        self._save_selector_episode_artifact(t_env, episode_num, selector_weights, module_scores, module_contributions, actions)

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        mean_return = rewards.sum().item() / max(mask_expanded.sum().item(), 1.0)
        self.latest_train_return = mean_return
        self.latest_selector_entropy = selector_entropy.item()
        self.latest_aux_reward_mean = auxiliary_reward.mean().item()
        self.latest_shaped_reward_delta_mean = clipped_shaped_reward_delta.mean().item()
        self.latest_shaped_reward_delta_clip_ratio = float((shaped_reward_delta.abs() > shaped_reward_delta_clip).float().mean().item())
        self.latest_selector_weight_mean = selector_weights.mean().item()
        self.latest_selector_advantage = 0.0
        self.latest_selector_balance = selector_balance.item()
        self.latest_selector_value_loss = float(selector_value_loss.item())
        self.latest_selector_trust_gate = float(trust_gate.mean().item())
        self.latest_selector_trust_bias_std = float(trust_bias.std(dim=-1).mean().item())
        self.latest_transition_loss = float(transition_loss.item())
        self.latest_transition_progress = float(transition_progress)
        observed_win = self.latest_test_win_rate if self.latest_test_win_rate > 0.0 else 0.0
        self.state_tracker.update(
            win=observed_win,
            episode_return=mean_return,
            td_loss=td_loss.item(),
            selector_entropy=selector_entropy.item(),
        )

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask_expanded.sum().item()
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("grad_norm", base_grad_norm, t_env)
            self.logger.log_stat("selector_grad_norm", selector_grad_norm, t_env)
            self.logger.log_stat("modular_reward_beta", self.modular_reward_beta, t_env)
            self.logger.log_stat("modular_reward_guidance_strength", guidance_strength, t_env)
            self.logger.log_stat("modular_reward_effective_beta", effective_beta, t_env)
            self.logger.log_stat("selector_mode", float(selector_mode), t_env)
            self.logger.log_stat("selector_entropy", self.latest_selector_entropy, t_env)
            self.logger.log_stat("shaped_reward_delta_mean", self.latest_shaped_reward_delta_mean, t_env)
            self.logger.log_stat("shaped_reward_delta_clip_ratio", self.latest_shaped_reward_delta_clip_ratio, t_env)
            self.logger.log_stat("selector_weight_mean", self.latest_selector_weight_mean, t_env)
            self.logger.log_stat("selector_weight_std", self.latest_selector_weight_std, t_env)
            self.logger.log_stat("selector_time_std", self.latest_selector_time_std, t_env)
            self.logger.log_stat("selector_agent_std", self.latest_selector_agent_std, t_env)
            self.logger.log_stat("selector_top1_switch_rate", self.latest_selector_top1_switch_rate, t_env)
            self.logger.log_stat("selector_advantage", self.latest_selector_advantage, t_env)
            self.logger.log_stat("selector_balance", self.latest_selector_balance, t_env)
            self.logger.log_stat("selector_value_loss", self.latest_selector_value_loss, t_env)
            self.logger.log_stat("selector_trust_gate", self.latest_selector_trust_gate, t_env)
            self.logger.log_stat("selector_trust_bias_std", self.latest_selector_trust_bias_std, t_env)
            self.logger.log_stat("selector_feedback_loss", self.latest_selector_feedback_loss, t_env)
            self.logger.log_stat("selector_feedback_value_loss", self.latest_selector_feedback_value_loss, t_env)
            self.logger.log_stat("selector_feedback_advantage", self.latest_selector_feedback_advantage, t_env)
            self.logger.log_stat("selector_feedback_target", self.latest_selector_feedback_target, t_env)
            self.logger.log_stat("transition_loss", self.latest_transition_loss, t_env)
            self.logger.log_stat("transition_progress", self.latest_transition_progress, t_env)
            self.logger.log_stat("outer_selector_loss", self.latest_outer_selector_loss, t_env)
            self.logger.log_stat("outer_selector_policy_loss", self.latest_outer_selector_policy_loss, t_env)
            self.logger.log_stat("outer_selector_value_loss", self.latest_outer_selector_value_loss, t_env)
            self.logger.log_stat("outer_selector_entropy", self.latest_outer_selector_entropy, t_env)
            self.logger.log_stat("outer_selector_advantage", self.latest_outer_selector_advantage, t_env)
            self.logger.log_stat("outer_selector_return_delta", self.latest_outer_selector_return_delta, t_env)
            self.logger.log_stat("outer_selector_return_delta_norm", self.latest_outer_selector_return_delta_norm, t_env)
            self.logger.log_stat("outer_selector_value", self.latest_outer_selector_value, t_env)
            self.logger.log_stat("outer_selector_alpha_mean", self.latest_outer_selector_alpha_mean, t_env)
            self.logger.log_stat("outer_selector_alpha_max", self.latest_outer_selector_alpha_max, t_env)
            if self.current_outer_prior is not None:
                for module_index, module_name in enumerate(self.module_pool.module_names):
                    self.logger.log_stat("outer_selector_prior_{}".format(module_name), float(self.current_outer_prior[module_index].item()), t_env)
            for module_name, module_score in self.module_pool.last_module_scores.items():
                self.logger.log_stat("module_score_{}".format(module_name), module_score, t_env)
            for module_name, module_usage in self.module_pool.last_module_usage.items():
                self.logger.log_stat("module_usage_{}".format(module_name), module_usage, t_env)
            for module_name, module_contribution in self.module_pool.last_module_contributions.items():
                self.logger.log_stat("module_contribution_{}".format(module_name), module_contribution, t_env)
            for module_name, module_weighted_score in self.module_pool.last_module_weighted_scores.items():
                self.logger.log_stat("module_weighted_score_{}".format(module_name), module_weighted_score, t_env)
            for module_name, module_activation in self.module_pool.last_module_activation.items():
                self.logger.log_stat("module_activation_{}".format(module_name), module_activation, t_env)
            for module_name, module_status_code in self.module_pool.last_module_status_code.items():
                self.logger.log_stat("module_status_{}".format(module_name), module_status_code, t_env)
            for module_name, module_scale in self.module_pool.last_module_scale.items():
                self.logger.log_stat("module_scale_{}".format(module_name), module_scale, t_env)
            for module_name, module_credit in self.module_pool.last_module_env_credit.items():
                self.logger.log_stat("module_env_credit_{}".format(module_name), module_credit, t_env)
            for module_name, module_trust in self.module_pool.last_module_trust.items():
                self.logger.log_stat("module_trust_{}".format(module_name), module_trust, t_env)
            for module_name, module_trust_prior in self.module_pool.last_module_trust_prior.items():
                self.logger.log_stat("module_trust_prior_{}".format(module_name), module_trust_prior, t_env)
            self.logger.log_stat("env_feedback_signal", self.latest_env_feedback_signal, t_env)
            self.logger.log_stat("module_update_event_count", self.latest_module_update_event_count, t_env)
            self.logger.log_stat("module_update_trigger_count", self.latest_module_update_trigger_count, t_env)
            trigger_decision = getattr(self.module_pool, "last_trigger_decision", {}) or {}
            trigger_mode_map = {
                "init": -1.0,
                "no_trigger": 0.0,
                "absolute_low_signal": 1.0,
                "relative_low_signal": 2.0,
                "relative_low_signal_improving": 3.0,
                "persistent_bad_module": 4.0,
            }
            self.logger.log_stat("module_update_trigger_mode", trigger_mode_map.get(trigger_decision.get("mode"), 0.0), t_env)
            self.logger.log_stat("module_update_trigger_stagnant", 1.0 if trigger_decision.get("stagnant", False) else 0.0, t_env)
            self.logger.log_stat("module_update_trigger_collapse", 1.0 if trigger_decision.get("collapse", False) else 0.0, t_env)
            self.logger.log_stat("module_update_trigger_improving", 1.0 if trigger_decision.get("improving", False) else 0.0, t_env)
            self.logger.log_stat("module_update_trigger_structural_bias", 1.0 if trigger_decision.get("structural_bias", False) else 0.0, t_env)
            self.logger.log_stat("module_update_trigger_persistent_bad", 1.0 if trigger_decision.get("persistent_bad", False) else 0.0, t_env)
            self.logger.log_stat("module_count", float(self.module_pool.num_modules()), t_env)
            self.logger.log_stat("module_pending_update", 1.0 if self.module_pool.pending_update is not None else 0.0, t_env)
            for module_name, transition_alpha in self.module_pool.last_transition_alpha.items():
                self.logger.log_stat("module_transition_alpha_{}".format(module_name), transition_alpha, t_env)
            for trace_name, trace_value in self.latest_selector_trace.items():
                self.logger.log_stat(trace_name, trace_value, t_env)
            self.logger.log_stat("aux_reward_mean", self.latest_aux_reward_mean, t_env)
            self.logger.log_stat("ivf_loss", self.latest_ivf_loss, t_env)
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask_expanded).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask_expanded).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _guidance_strength(self, t_env: int) -> float:
        if self.modular_reward_guidance_ramp_steps <= 0:
            return 1.0
        ramp_start = max(self.modular_reward_guidance_start_steps, 0)
        if t_env <= ramp_start:
            return self.modular_reward_min_beta_ratio
        progress = min(max((t_env - ramp_start) / float(max(self.modular_reward_guidance_ramp_steps, 1)), 0.0), 1.0)
        return self.modular_reward_min_beta_ratio + (1.0 - self.modular_reward_min_beta_ratio) * progress

    def _effective_beta(self, t_env: int) -> float:
        ramp_steps = max(int(self.modular_reward_beta_ramp_steps), 0)
        ramp_start = max(int(self.modular_reward_beta_ramp_start_steps), 0)
        if ramp_steps <= 0:
            return float(self.modular_reward_beta_end)
        if t_env <= ramp_start:
            return float(self.modular_reward_beta_start)
        progress = min(max((t_env - ramp_start) / float(max(ramp_steps, 1)), 0.0), 1.0)
        return float(self.modular_reward_beta_start) + (float(self.modular_reward_beta_end) - float(self.modular_reward_beta_start)) * progress

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _accumulate_selector_feedback_stats(self, selector_context, module_contributions):
        mean_context = selector_context.detach().mean(dim=(0, 1, 2))
        mean_exposure = module_contributions.detach().mean(dim=(0, 1, 2))
        if self._selector_feedback_context_sum is None:
            self._selector_feedback_context_sum = mean_context.clone()
            self._selector_feedback_exposure_sum = mean_exposure.clone()
        else:
            self._selector_feedback_context_sum = self._selector_feedback_context_sum + mean_context
            self._selector_feedback_exposure_sum = self._selector_feedback_exposure_sum + mean_exposure
        self._selector_feedback_batches += 1

    def _consume_selector_feedback_summary(self):
        if self._selector_feedback_batches <= 0 or self._selector_feedback_context_sum is None or self._selector_feedback_exposure_sum is None:
            return None
        summary = {
            "context": (self._selector_feedback_context_sum / float(self._selector_feedback_batches)).detach().cpu(),
            "exposure": (self._selector_feedback_exposure_sum / float(self._selector_feedback_batches)).detach().cpu(),
            "count": int(self._selector_feedback_batches),
        }
        self._selector_feedback_context_sum = None
        self._selector_feedback_exposure_sum = None
        self._selector_feedback_batches = 0
        return summary

    def _update_selector_feedback_head(self, selector_summary, trust_update):
        if selector_summary is None:
            return
        if "return_delta" not in trust_update or "win_rate_delta" not in trust_update:
            return

        context = selector_summary["context"].to(next(self.selector_feedback_net.parameters()).device)
        exposure = selector_summary["exposure"].to(context.device)
        if context.numel() <= 0 or exposure.numel() <= 0:
            return

        return_delta = float(trust_update.get("return_delta", 0.0))
        win_rate_delta = float(trust_update.get("win_rate_delta", 0.0))
        return_scale = max(float(getattr(self.args, "modular_reward_env_trust_return_scale", 2.0)), 1e-6)
        win_rate_scale = max(float(getattr(self.args, "modular_reward_env_trust_win_scale", 0.05)), 1e-6)
        normalized_return = max(min(return_delta / return_scale, 1.0), -1.0)
        normalized_win = max(min(win_rate_delta / win_rate_scale, 1.0), -1.0)
        target_signal = 0.7 * normalized_return + 0.3 * normalized_win

        exposure = exposure - exposure.mean()
        exposure_std = exposure.std(unbiased=False).clamp(min=1e-4)
        target_logits = (exposure / exposure_std) * self.selector_feedback_target_temperature * target_signal
        trust_prior = trust_update.get("prior", [])
        if trust_prior:
            trust_prior_tensor = th.tensor(trust_prior, dtype=th.float32, device=context.device)
            trust_prior_tensor = trust_prior_tensor / trust_prior_tensor.sum().clamp(min=1e-8)
            trust_logits = th.log(trust_prior_tensor.clamp(min=1e-8)) - math.log(1.0 / max(trust_prior_tensor.numel(), 1))
            target_logits = target_logits + self.selector_feedback_trust_prior_coef * trust_logits
        target_prior = th.softmax(target_logits, dim=0)

        feedback_logits = self.selector_feedback_net(context.view(1, 1, 1, -1)).view(-1)
        feedback_policy = th.softmax(feedback_logits, dim=-1)
        log_feedback_policy = th.log(feedback_policy.clamp(min=1e-8))
        feedback_value = self.selector_feedback_value_net(context.view(1, 1, 1, -1)).view(-1)[0]
        feedback_target = th.tensor(target_signal, dtype=th.float32, device=context.device)
        feedback_advantage = feedback_target - feedback_value.detach()
        policy_loss = -(target_prior.detach() * log_feedback_policy).sum() * feedback_advantage
        value_loss = F.mse_loss(feedback_value, feedback_target)
        entropy = -(feedback_policy * log_feedback_policy).sum()
        feedback_loss = policy_loss + self.selector_feedback_value_coef * value_loss - self.selector_entropy_coef * entropy

        self.selector_optimizer.zero_grad()
        feedback_loss.backward()
        th.nn.utils.clip_grad_norm_(self.selector_params, self.args.grad_norm_clip)
        self.selector_optimizer.step()

        self.latest_selector_feedback_loss = float(policy_loss.item())
        self.latest_selector_feedback_value_loss = float(value_loss.item())
        self.latest_selector_feedback_advantage = float(feedback_advantage.item())
        self.latest_selector_feedback_target = float(target_signal)

    def _record_module_stats(self, selector_weights, module_scores, module_contributions):
        activation_mask = (selector_weights >= self.selector_activation_threshold).float()
        for index, spec in enumerate(self.module_pool.module_specs):
            module_id = spec["id"]
            self.module_pool.last_module_usage[module_id] = float(selector_weights[..., index].mean().item())
            self.module_pool.last_module_scores[module_id] = float(module_scores[..., index].mean().item())
            self.module_pool.last_module_contributions[module_id] = float(module_contributions[..., index].mean().item())
            self.module_pool.last_module_weighted_scores[module_id] = float(
                (selector_weights[..., index] * module_scores[..., index]).mean().item()
            )
            self.module_pool.last_module_activation[module_id] = float(activation_mask[..., index].mean().item())

    def _record_selector_diagnostics(self, selector_weights):
        self.latest_selector_weight_std = float(selector_weights.std(dim=-1).mean().item())
        if selector_weights.size(1) > 1:
            self.latest_selector_time_std = float(selector_weights.std(dim=1).mean().item())
            top1_indices = selector_weights.argmax(dim=-1)
            top1_switches = (top1_indices[:, 1:] != top1_indices[:, :-1]).float()
            self.latest_selector_top1_switch_rate = float(top1_switches.mean().item())
        else:
            self.latest_selector_time_std = 0.0
            self.latest_selector_top1_switch_rate = 0.0

        if selector_weights.size(2) > 1:
            self.latest_selector_agent_std = float(selector_weights.std(dim=2).mean().item())
        else:
            self.latest_selector_agent_std = 0.0

        self.latest_selector_trace = {}
        sample_weights = selector_weights[0, :min(selector_weights.size(1), 3), 0]
        for step_index in range(sample_weights.size(0)):
            for module_index, module_name in enumerate(self.module_pool.module_names):
                trace_key = "selector_trace_t{}_{}".format(step_index, module_name)
                self.latest_selector_trace[trace_key] = float(sample_weights[step_index, module_index].item())

    def _save_selector_episode_artifact(self, t_env, episode_num, selector_weights, module_scores, module_contributions, actions):
        interval = int(getattr(self.args, "modular_reward_selector_viz_interval", 0))
        if interval <= 0 or t_env <= 0 or t_env - self.latest_selector_viz_t_env < interval:
            return
        self.latest_selector_viz_t_env = t_env
        base_dir = os.path.join(self.args.local_results_path, "selector_episode_artifacts")
        os.makedirs(base_dir, exist_ok=True)
        payload = {
            "t_env": int(t_env),
            "episode_num": int(episode_num),
            "module_ids": [spec["id"] for spec in self.module_pool.module_specs],
            "module_names": [spec["name"] for spec in self.module_pool.module_specs],
            "selector_weights": selector_weights[0].detach().cpu().tolist(),
            "module_scores": module_scores[0].detach().cpu().tolist(),
            "module_contributions": module_contributions[0].detach().cpu().tolist(),
            "actions": actions[0].detach().cpu().squeeze(-1).tolist(),
        }
        path = os.path.join(base_dir, "{}_{}.json".format(getattr(self.args, "unique_token", "selector_episode"), int(t_env)))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def maybe_refine_modules(self, t_env: int):
        self.maybe_update_outer_loop_selector(t_env)

        feedback_payload = {
            "t_env": t_env,
            "latest_test_return": self._last_eval_return,
            "latest_test_win_rate": self.latest_test_win_rate,
            "latest_train_return": self.latest_train_return,
            "selector_entropy": self.latest_selector_entropy,
            "selector_advantage": self.latest_selector_advantage,
            "selector_balance": self.latest_selector_balance,
            "selector_value_loss": self.latest_selector_value_loss,
            "selector_top1_switch_rate": self.latest_selector_top1_switch_rate,
            "aux_reward_mean": self.latest_aux_reward_mean,
            "outer_selector_loss": self.latest_outer_selector_loss,
            "outer_selector_return_delta": self.latest_outer_selector_return_delta,
            "outer_selector_alpha_mean": self.latest_outer_selector_alpha_mean,
        }
        feedback_payload.update(self.module_pool.get_latest_stats())
        self.module_pool.record_window_metrics(t_env, feedback_payload=feedback_payload)
        selector_summary = self._consume_selector_feedback_summary()
        trust_update = self.module_pool.update_env_feedback_trust(feedback_payload, t_env)
        self._update_selector_feedback_head(selector_summary, trust_update)
        self.latest_env_feedback_signal = float(trust_update.get("signal", 0.0))

        if self.modular_reward_module_update_interval > 0 and t_env > 0 and t_env - self.latest_module_update_check_t_env >= self.modular_reward_module_update_interval:
            self.latest_module_update_check_t_env = t_env
            update_events = self.module_pool.apply_feedback_update(feedback_payload, t_env)
            self.latest_module_update_event_count = float(len(update_events))
            self.latest_module_update_trigger_count = 1.0 if update_events else 0.0
            if update_events:
                self.module_pool.save_specs(
                    "{}_modules_{}".format(getattr(self.args, "unique_token", "modular_reward_update"), t_env),
                    self.args.local_results_path,
                )
                self.logger.console_logger.info("Applied modular reward module update at t_env {}: {}".format(t_env, update_events))
        else:
            self.latest_module_update_event_count = 0.0
            self.latest_module_update_trigger_count = 0.0

        refinement_interval = getattr(self.args, "modular_reward_refinement_interval", 0)
        if refinement_interval <= 0 or t_env <= 0 or t_env % refinement_interval != 0:
            return
        if not getattr(self.args, "modular_reward_use_llm_refinement", False):
            return

        current_specs = self.module_pool.get_specs_payload()
        refined_scales = self.module_pool.initializer.refine_scales(current_specs, feedback_payload)
        if refined_scales:
            self.module_pool.update_scales(refined_scales, t_env=t_env, source="llm_refinement")
            self.module_pool.save_specs(getattr(self.args, "unique_token", "modular_reward_refined"), self.args.local_results_path)
            self.logger.console_logger.info("Applied modular reward scale refinement at t_env {}".format(t_env))

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.selector_net.cuda()
        self.selector_feedback_net.cuda()
        self.selector_value_net.cuda()
        self.selector_feedback_value_net.cuda()
        self.selector_trust_gate_net.cuda()
        self.outer_loop_selector.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.selector_net.state_dict(), "{}/selector.th".format(path))
        th.save(self.selector_feedback_net.state_dict(), "{}/selector_feedback.th".format(path))
        th.save(self.selector_value_net.state_dict(), "{}/selector_value.th".format(path))
        th.save(self.selector_feedback_value_net.state_dict(), "{}/selector_feedback_value.th".format(path))
        th.save(self.selector_trust_gate_net.state_dict(), "{}/selector_trust_gate.th".format(path))
        th.save(self.outer_loop_selector.policy.state_dict(), "{}/outer_selector_policy.th".format(path))
        th.save(self.outer_loop_selector.value.state_dict(), "{}/outer_selector_value.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.selector_optimizer.state_dict(), "{}/selector_opt.th".format(path))
        th.save(self.outer_loop_selector.optimizer.state_dict(), "{}/outer_selector_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        selector_path = "{}/selector.th".format(path)
        if os.path.exists(selector_path):
            self.selector_net.load_state_dict(th.load(selector_path, map_location=lambda storage, loc: storage))
        selector_feedback_path = "{}/selector_feedback.th".format(path)
        if os.path.exists(selector_feedback_path):
            self.selector_feedback_net.load_state_dict(th.load(selector_feedback_path, map_location=lambda storage, loc: storage))
        selector_value_path = "{}/selector_value.th".format(path)
        if os.path.exists(selector_value_path):
            self.selector_value_net.load_state_dict(th.load(selector_value_path, map_location=lambda storage, loc: storage))
        selector_feedback_value_path = "{}/selector_feedback_value.th".format(path)
        if os.path.exists(selector_feedback_value_path):
            self.selector_feedback_value_net.load_state_dict(th.load(selector_feedback_value_path, map_location=lambda storage, loc: storage))
        selector_trust_gate_path = "{}/selector_trust_gate.th".format(path)
        if os.path.exists(selector_trust_gate_path):
            self.selector_trust_gate_net.load_state_dict(th.load(selector_trust_gate_path, map_location=lambda storage, loc: storage))
        outer_selector_policy_path = "{}/outer_selector_policy.th".format(path)
        if os.path.exists(outer_selector_policy_path):
            self.outer_loop_selector.policy.load_state_dict(th.load(outer_selector_policy_path, map_location=lambda storage, loc: storage))
        outer_selector_value_path = "{}/outer_selector_value.th".format(path)
        if os.path.exists(outer_selector_value_path):
            self.outer_loop_selector.value.load_state_dict(th.load(outer_selector_value_path, map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        selector_opt_path = "{}/selector_opt.th".format(path)
        if os.path.exists(selector_opt_path):
            self.selector_optimizer.load_state_dict(th.load(selector_opt_path, map_location=lambda storage, loc: storage))
        outer_selector_opt_path = "{}/outer_selector_opt.th".format(path)
        if os.path.exists(outer_selector_opt_path):
            self.outer_loop_selector.optimizer.load_state_dict(th.load(outer_selector_opt_path, map_location=lambda storage, loc: storage))
