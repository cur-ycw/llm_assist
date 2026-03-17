import copy
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer


class SCAREQLearner:
    """
    SCARE QLearner: Dual-track optimization for SCARE architecture.

    Track 1 (TD): Standard QMIX TD-error loss -> updates fc2 + mixer (NOT encoders, NOT selector)
    Track 2 (Selector): Routing loss + Prior loss -> updates ONLY the SelectorMLP

    Key constraints:
    - Encoders are ALWAYS frozen (eval mode, no grad)
    - Selector gradients come ONLY from L_routing + L_prior, never from TD loss
    - Counterfactual search: for agent i, fix Q_{-i}, enumerate a'_i, compute Q_tot via mixer
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        # --- Freeze encoders ---
        self.mac.agent.freeze_encoders()

        # --- Track 1: TD params (fc2 only, NOT selector, NOT encoders) ---
        self.td_params = list(self.mac.agent.fc2.parameters())

        # --- Mixer ---
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.td_params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.td_optimiser = RMSprop(
            params=self.td_params, lr=args.lr,
            alpha=args.optim_alpha, eps=args.optim_eps
        )

        # --- Track 2: Selector params ---
        self.selector_params = list(self.mac.agent.selector.parameters())
        selector_lr = getattr(args, "selector_lr", 0.0005)
        self.selector_optimiser = RMSprop(
            params=self.selector_params, lr=selector_lr,
            alpha=args.optim_alpha, eps=args.optim_eps
        )

        # --- Prior loss decay schedule ---
        self.lambda_prior_start = getattr(args, "lambda_prior_start", 1.0)
        self.lambda_prior_finish = getattr(args, "lambda_prior_finish", 0.01)
        self.lambda_prior_anneal = getattr(args, "lambda_prior_anneal", 200000)

        # --- Target network ---
        self.target_mac = copy.deepcopy(mac)
        self.target_mac.agent.freeze_encoders()

        # --- Alpha prior (injected by run.py from pretrained alpha_prior.json) ---
        self.alpha_prior = None
        if hasattr(args, 'alpha_prior') and args.alpha_prior:
            self._build_alpha_prior(args.alpha_prior)

        self.last_target_update_episode = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

    def _get_lambda_prior(self, t_env):
        """Linear decay of prior loss weight."""
        delta = (self.lambda_prior_start - self.lambda_prior_finish) / self.lambda_prior_anneal
        return max(self.lambda_prior_finish, self.lambda_prior_start - delta * t_env)

    def _build_alpha_prior(self, prior_dict):
        """Convert alpha_prior.json dict to tensor (K, K)."""
        K = getattr(self.args, 'n_encoders', 4)
        prior = th.ones(K, K) / K  # fallback uniform
        for qi, scores in prior_dict.items():
            qi_idx = int(qi) - 1  # 1-indexed -> 0-indexed
            for pi, val in scores.items():
                pi_idx = int(pi) - 1
                if 0 <= qi_idx < K and 0 <= pi_idx < K:
                    prior[qi_idx, pi_idx] = float(val)
        # Normalize rows
        prior = prior / prior.sum(dim=-1, keepdim=True)
        self.alpha_prior = prior  # (K, K)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # ============================================================
        # Forward pass: collect Q-values AND alpha for all timesteps
        # ============================================================
        mac_out = []
        alpha_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            # Retrieve alpha stored by BasicMAC during forward
            if hasattr(self.mac, 'selector_alpha') and self.mac.selector_alpha is not None:
                alpha_t = self.mac.selector_alpha.view(
                    batch.batch_size, self.n_agents, -1
                )
                alpha_out.append(alpha_t)

        mac_out = th.stack(mac_out, dim=1)  # (bs, T, n_agents, n_actions)

        if alpha_out:
            alpha_out = th.stack(alpha_out, dim=1)  # (bs, T, n_agents, K)

        # Pick Q-values for taken actions
        chosen_action_qvals = th.gather(
            mac_out[:, :-1], dim=3, index=actions
        ).squeeze(3)  # (bs, T-1, n_agents)

        # ============================================================
        # Target network forward pass
        # ============================================================
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        target_mac_out = th.stack(target_mac_out[1:], dim=1)
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Double Q-learning
        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals_mixed = self.mixer(
                chosen_action_qvals, batch["state"][:, :-1]
            )
            target_max_qvals = self.target_mixer(
                target_max_qvals, batch["state"][:, 1:]
            )
        else:
            chosen_action_qvals_mixed = chosen_action_qvals

        # ============================================================
        # Track 1: TD Loss (updates fc2 + mixer)
        # ============================================================
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        td_error = chosen_action_qvals_mixed - targets.detach()
        td_mask = mask.expand_as(td_error)
        masked_td_error = td_error * td_mask
        td_loss = (masked_td_error ** 2).sum() / td_mask.sum()

        self.td_optimiser.zero_grad()
        td_loss.backward()
        td_grad_norm = th.nn.utils.clip_grad_norm_(
            self.td_params, self.args.grad_norm_clip
        )
        self.td_optimiser.step()

        # ============================================================
        # Track 2: Selector Loss (updates ONLY selector)
        # ============================================================
        selector_loss = th.tensor(0.0, device=batch.device)
        routing_loss = th.tensor(0.0, device=batch.device)
        prior_loss = th.tensor(0.0, device=batch.device)

        if alpha_out is not None and len(alpha_out) > 0:
            selector_loss, routing_loss, prior_loss = self._compute_selector_loss(
                batch, mac_out, alpha_out, mask, t_env
            )

            self.selector_optimiser.zero_grad()
            selector_loss.backward()
            sel_grad_norm = th.nn.utils.clip_grad_norm_(
                self.selector_params, self.args.grad_norm_clip
            )
            self.selector_optimiser.step()

        # ============================================================
        # Target network update
        # ============================================================
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # ============================================================
        # Logging
        # ============================================================
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("td_grad_norm", td_grad_norm, t_env)
            self.logger.log_stat("selector_loss", selector_loss.item(), t_env)
            self.logger.log_stat("routing_loss", routing_loss.item(), t_env)
            self.logger.log_stat("prior_loss", prior_loss.item(), t_env)
            self.logger.log_stat("lambda_prior", self._get_lambda_prior(t_env), t_env)
            mask_elems = td_mask.sum().item()
            self.logger.log_stat(
                "td_error_abs",
                masked_td_error.abs().sum().item() / mask_elems, t_env
            )
            self.log_stats_t = t_env

    def _compute_selector_loss(self, batch, mac_out, alpha_out, mask, t_env):
        """
        Compute the two selector losses using a FRESH forward pass.
        The original alpha_out from the first forward pass is stale after td_loss.backward(),
        so we do a fresh forward to get new alpha with live gradients.

        1. L_routing: counterfactual value gap
        2. L_prior: KL divergence with LLM tactical prior
        """
        bs = batch.batch_size

        # --- Fresh forward pass (shared by both losses) ---
        fresh_mac_out = []
        fresh_alpha_out = []
        self.mac.init_hidden(bs)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            fresh_mac_out.append(agent_outs)
            if hasattr(self.mac, 'selector_alpha') and self.mac.selector_alpha is not None:
                alpha_t = self.mac.selector_alpha.view(bs, self.n_agents, -1)
                fresh_alpha_out.append(alpha_t)

        fresh_mac_out = th.stack(fresh_mac_out, dim=1)  # (bs, T+1, n_agents, n_actions)
        if fresh_alpha_out:
            fresh_alpha_out = th.stack(fresh_alpha_out, dim=1)  # (bs, T+1, n_agents, K)

        # ---- Routing Loss (Counterfactual Value Gap) ----
        routing_loss = self._compute_routing_loss_from_fresh(
            batch, fresh_mac_out, mask
        )

        # ---- Prior Loss (KL divergence) using fresh alpha ----
        if fresh_alpha_out is not None and len(fresh_alpha_out) > 0:
            prior_loss = self._compute_prior_loss(fresh_alpha_out, batch, mask)
        else:
            prior_loss = th.tensor(0.0, device=batch.device)

        lambda_prior = self._get_lambda_prior(t_env)
        total = routing_loss + lambda_prior * prior_loss
        return total, routing_loss, prior_loss

    def _compute_routing_loss(self, batch, mac_out, mask):
        """Legacy wrapper — delegates to fresh-pass version."""
        return self._compute_routing_loss_from_fresh(batch, None, mask, do_fresh=True)

    def _compute_routing_loss_from_fresh(self, batch, fresh_mac_out, mask, do_fresh=False):
        """
        Counterfactual value gap using pre-computed fresh forward pass.
        """
        if self.mixer is None:
            return th.tensor(0.0, device=batch.device)

        bs = batch.batch_size
        states = batch["state"][:, :-1]
        actions = batch["actions"][:, :-1]

        # If no fresh_mac_out provided, do a fresh forward pass
        if fresh_mac_out is None or do_fresh:
            fresh_mac_out = []
            self.mac.init_hidden(bs)
            for t in range(batch.max_seq_length):
                agent_outs = self.mac.forward(batch, t=t)
                fresh_mac_out.append(agent_outs)
            fresh_mac_out = th.stack(fresh_mac_out, dim=1)

        T = fresh_mac_out.size(1) - 1
        fresh_q = fresh_mac_out[:, :-1]

        # Q values for taken actions
        chosen_q = th.gather(fresh_q, dim=3, index=actions).squeeze(3)  # (bs, T, n_agents)

        # Q_tot for the real actions
        q_tot_real = self.mixer(chosen_q, states)  # (bs, T, 1)

        # --- Counterfactual: for each agent i, try all actions ---
        # Detach other agents' Q to fix Q_{-i}
        chosen_q_detached = chosen_q.detach()  # (bs, T, n_agents)

        cf_max_qtot = th.zeros_like(q_tot_real)  # (bs, T, 1)

        for i in range(self.n_agents):
            # For agent i, enumerate all possible actions
            q_i_all = fresh_q[:, :, i, :]  # (bs, T, n_actions) - has selector grad

            best_qtot_i = th.full(
                (bs, T, 1), -float('inf'), device=batch.device
            )

            for a in range(self.n_actions):
                # Replace agent i's Q with Q(a)
                q_a = q_i_all[:, :, a]  # (bs, T)
                # Build counterfactual joint Q vector
                cf_chosen = chosen_q_detached.clone()  # (bs, T, n_agents)
                cf_chosen[:, :, i] = q_a  # only agent i changes

                # Compute Q_tot through mixer (detach mixer to avoid TD interference)
                cf_qtot = self.mixer(cf_chosen, states)  # (bs, T, 1)
                best_qtot_i = th.max(best_qtot_i, cf_qtot)

            cf_max_qtot = cf_max_qtot + best_qtot_i

        # Average over agents
        cf_max_qtot = cf_max_qtot / self.n_agents

        # Value gap loss: (Q_tot_real - max_potential)^2
        gap = q_tot_real - cf_max_qtot.detach()
        gap_mask = mask.expand_as(gap)
        routing_loss = ((gap * gap_mask) ** 2).sum() / gap_mask.sum()

        return routing_loss

    def _compute_prior_loss(self, alpha_out, batch, mask):
        """
        KL divergence between selector's alpha and LLM tactical prior alpha_LLM.
        L_prior = KL(alpha || alpha_LLM)

        If no prior is provided, uses uniform distribution as fallback.
        """
        # alpha_out: (bs, T+1, n_agents, K)
        alpha = alpha_out[:, :-1]  # (bs, T, n_agents, K)
        K = alpha.size(-1)

        # Use pretrained LLM prior if available, fallback to uniform
        if self.alpha_prior is not None:
            # alpha_prior: (K, K) -> average over query dim -> (K,)
            alpha_prior = self.alpha_prior.mean(dim=0).to(alpha.device)
            # Broadcast to (bs, T, n_agents, K)
            alpha_prior = alpha_prior.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(alpha)
        else:
            alpha_prior = th.ones_like(alpha) / K  # uniform

        # KL(alpha || alpha_prior) = sum(alpha * log(alpha / alpha_prior))
        # Use F.kl_div which expects log-prob as input
        log_alpha = th.log(alpha + 1e-8)
        log_prior = th.log(alpha_prior + 1e-8)

        # F.kl_div(input=log_q, target=p) computes sum(p * (log_p - log_q))
        # We want KL(alpha || prior) = sum(alpha * (log_alpha - log_prior))
        kl = (alpha * (log_alpha - log_prior)).sum(dim=-1)  # (bs, T, n_agents)

        # Average over agents
        kl = kl.mean(dim=-1, keepdim=True)  # (bs, T, 1)

        kl_mask = mask.expand_as(kl)
        prior_loss = (kl * kl_mask).sum() / kl_mask.sum()

        return prior_loss

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.td_optimiser.state_dict(), "{}/td_opt.th".format(path))
        th.save(self.selector_optimiser.state_dict(), "{}/sel_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage)
            )
        self.td_optimiser.load_state_dict(
            th.load("{}/td_opt.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.selector_optimiser.load_state_dict(
            th.load("{}/sel_opt.th".format(path), map_location=lambda storage, loc: storage)
        )
