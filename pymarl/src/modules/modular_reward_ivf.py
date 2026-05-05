from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainingStateTracker:
    def __init__(self, window=200):
        self.window = window
        self.win_history = deque(maxlen=window)
        self.return_history = deque(maxlen=window)
        self.loss_history = deque(maxlen=window * 5)
        self.selector_entropy_history = deque(maxlen=window)
        self.last_action = 0
        self.steps_since = 0
        self.episode_count = 0

    def update(self, win: float, episode_return: float, td_loss: float, selector_entropy: float):
        self.win_history.append(float(win))
        self.return_history.append(float(episode_return))
        self.loss_history.append(float(td_loss))
        self.selector_entropy_history.append(float(selector_entropy))
        self.episode_count += 1

    def record_action(self, action: int):
        self.last_action = int(action)
        self.steps_since = 0

    def tick(self, n_steps: int = 1):
        self.steps_since += int(n_steps)

    def get_state(self) -> torch.Tensor:
        wins = list(self.win_history) if self.win_history else [0.0]
        returns = list(self.return_history) if self.return_history else [0.0]
        losses = list(self.loss_history) if self.loss_history else [1.0]
        entropies = list(self.selector_entropy_history) if self.selector_entropy_history else [1.0]

        win_rate = float(np.mean(wins))
        if len(wins) >= 10:
            x = np.arange(len(wins), dtype=np.float64)
            x = x - x.mean()
            y = np.array(wins, dtype=np.float64)
            y = y - y.mean()
            denom = float((x * x).sum())
            slope = float((x * y).sum() / denom) if denom > 0 else 0.0
        else:
            slope = 0.0

        mean_return = float(np.mean(returns))
        return_norm = np.clip(mean_return / max(abs(mean_return), 20.0), -1.0, 1.0)
        entropy = float(np.mean(entropies))

        if len(losses) >= 20:
            recent = np.mean(losses[-20:])
            older = np.mean(losses[-40:-20] if len(losses) >= 40 else losses[:20])
            plateau = 1.0 if abs(recent - older) < 0.01 * max(abs(older), 1e-6) else 0.0
        else:
            plateau = 0.0

        return torch.tensor([
            win_rate,
            np.clip(slope * 100.0, -1.0, 1.0),
            return_norm,
            np.clip(entropy, 0.0, 2.0) / 2.0,
            plateau,
            self.last_action / 2.0,
            min(self.steps_since / 500000.0, 1.0),
            min(self.episode_count / 10000.0, 1.0),
        ], dtype=torch.float32)


class SelectorNetwork(nn.Module):
    def __init__(self, input_dim, n_modules, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_modules),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        original_shape = context.shape[:-1]
        flat_context = context.reshape(-1, context.shape[-1])
        logits = self.net(flat_context)
        return logits.reshape(*original_shape, logits.shape[-1])


class ContextValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        original_shape = context.shape[:-1]
        flat_context = context.reshape(-1, context.shape[-1])
        values = self.net(flat_context)
        return values.reshape(*original_shape, 1)


class ModulePriorNetwork(nn.Module):
    def __init__(self, input_dim, n_modules, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_modules),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        original_shape = state.shape[:-1]
        flat_state = state.reshape(-1, state.shape[-1])
        logits = self.net(flat_state)
        return logits.reshape(*original_shape, logits.shape[-1])


class OuterLoopRewardSelector:
    def __init__(
        self,
        state_dim=8,
        n_modules=4,
        hidden_dim=64,
        lr=5e-4,
        entropy_coef=0.01,
        value_coef=0.5,
        min_alpha=0.3,
        return_scale=20.0,
    ):
        self.policy = ModulePriorNetwork(state_dim, n_modules, hidden_dim=hidden_dim)
        self.value = ContextValueNetwork(state_dim, hidden_dim=hidden_dim)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr,
        )
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.min_alpha = min_alpha
        self.return_scale = max(return_scale, 1e-6)

    def _device(self):
        return next(self.policy.parameters()).device

    def _distribution(self, state: torch.Tensor):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        logits = self.policy(state)
        alpha = F.softplus(logits) + self.min_alpha
        dist = torch.distributions.Dirichlet(alpha)
        values = self.value(state).squeeze(-1)
        return dist, alpha, values

    def sample_prior(self, state: torch.Tensor):
        with torch.no_grad():
            state = state.to(self._device())
            dist, alpha, values = self._distribution(state)
            prior = dist.sample().squeeze(0)
            return {
                "prior": prior.detach().cpu(),
                "alpha": alpha.squeeze(0).detach().cpu(),
                "value": float(values.squeeze(0).item()),
                "entropy": float(dist.entropy().squeeze(0).item()),
            }

    def update(self, state: torch.Tensor, sampled_prior: torch.Tensor, return_delta: float):
        state = state.to(self._device())
        sampled_prior = sampled_prior.to(self._device()).clamp(min=1e-6)
        dist, alpha, values = self._distribution(state)
        target = torch.tensor(
            np.clip(float(return_delta) / self.return_scale, -1.0, 1.0),
            dtype=torch.float32,
            device=self._device(),
        )
        value = values.squeeze(0)
        log_prob = dist.log_prob(sampled_prior.unsqueeze(0)).squeeze(0)
        entropy = dist.entropy().squeeze(0)
        advantage = target - value.detach()
        policy_loss = -log_prob * advantage
        value_loss = F.mse_loss(value, target)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "advantage": float(advantage.item()),
            "return_delta": float(return_delta),
            "return_delta_norm": float(target.item()),
            "alpha_mean": float(alpha.mean().item()),
            "alpha_max": float(alpha.max().item()),
            "value": float(value.item()),
        }

    def cuda(self):
        self.policy.cuda()
        self.value.cuda()


class InterventionValueFunction(nn.Module):
    def __init__(self, state_dim=8, n_actions=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class IVFTrainer:
    def __init__(self, n_actions=3, lr=1e-3, gamma=0.5, epsilon_start=0.5, epsilon_end=0.1, epsilon_decay_steps=80):
        self.ivf = InterventionValueFunction(n_actions=n_actions)
        self.optimizer = torch.optim.Adam(self.ivf.parameters(), lr=lr)
        self.replay = deque(maxlen=500)
        self.gamma = gamma
        self.n_actions = n_actions
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / max(epsilon_decay_steps, 1)

    def select_action(self, state: torch.Tensor) -> int:
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        with torch.no_grad():
            return int(self.ivf(state).argmax().item())

    def store(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor):
        self.replay.append((state.detach().clone(), int(action), float(reward), next_state.detach().clone()))

    def update(self, min_samples=8, batch_size=32) -> float:
        if len(self.replay) < min_samples:
            return 0.0

        indices = np.random.choice(len(self.replay), size=min(batch_size, len(self.replay)), replace=False)
        total_loss = 0.0
        for idx in indices:
            state, action, reward, next_state = self.replay[idx]
            q_values = self.ivf(state)
            q_current = q_values[action]
            with torch.no_grad():
                q_next = self.ivf(next_state).max()
                target = reward + self.gamma * q_next
            loss = F.mse_loss(q_current, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(indices)
