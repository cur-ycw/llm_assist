import torch as th
import torch.nn as nn
import torch.nn.functional as F
import copy


class EncoderBranch(nn.Module):
    """A single frozen encoder branch: fc1 -> GRU -> fc_enc."""

    def __init__(self, input_shape, rnn_hidden_dim, enc_feature_dim):
        super(EncoderBranch, self).__init__()
        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc_enc = nn.Linear(rnn_hidden_dim, enc_feature_dim)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h = self.rnn(x, hidden_state)
        feat = self.fc_enc(h)
        return feat, h


class SelectorMLP(nn.Module):
    """Trainable routing selector: obs -> alpha weights over K encoders."""

    def __init__(self, input_shape, hidden_dim, n_encoders):
        super(SelectorMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_encoders),
        )

    def forward(self, inputs):
        logits = self.net(inputs)
        alpha = F.softmax(logits, dim=-1)  # (batch*n_agents, K)
        return alpha


class SCARERNNAgent(nn.Module):
    """
    SCARE Agent: K frozen encoder branches + trainable selector + trainable head.

    Architecture:
        obs -> SelectorMLP -> alpha (K weights)
        obs -> EncoderBranch_k -> feat_k  (frozen, k=1..K)
        combined_feat = sum(alpha_k * feat_k)
        q = fc2(combined_feat)

    Returns: q, h, alpha
        - q: (batch*n_agents, n_actions)
        - h: (batch*n_agents, K, rnn_hidden_dim) stacked hidden states
        - alpha: (batch*n_agents, K) routing weights
    """

    def __init__(self, input_shape, args):
        super(SCARERNNAgent, self).__init__()
        self.args = args
        self.n_encoders = getattr(args, "n_encoders", 4)
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.enc_feature_dim = getattr(args, "enc_feature_dim", args.rnn_hidden_dim)

        # --- K Frozen Encoder Branches ---
        self.encoders = nn.ModuleList([
            EncoderBranch(input_shape, self.rnn_hidden_dim, self.enc_feature_dim)
            for _ in range(self.n_encoders)
        ])

        # --- Trainable Selector ---
        selector_hidden_dim = getattr(args, "selector_hidden_dim", 64)
        self.selector = SelectorMLP(input_shape, selector_hidden_dim, self.n_encoders)

        # --- Trainable Q-head ---
        self.fc2 = nn.Linear(self.enc_feature_dim, args.n_actions)

    def init_hidden(self):
        # Return zeros for K encoder hidden states: (1, K * rnn_hidden_dim)
        # We flatten K hidden states into one vector for compatibility with BasicMAC
        return self.fc2.weight.new(1, self.n_encoders * self.rnn_hidden_dim).zero_()

    def freeze_encoders(self):
        """Set all encoder branches to eval mode and disable gradients."""
        for enc in self.encoders:
            enc.eval()
            for p in enc.parameters():
                p.requires_grad = False

    def forward(self, inputs, hidden_state):
        """
        Args:
            inputs: (batch*n_agents, input_shape)
            hidden_state: (batch*n_agents, K * rnn_hidden_dim)

        Returns:
            q: (batch*n_agents, n_actions)
            h_out: (batch*n_agents, K * rnn_hidden_dim)
            alpha: (batch*n_agents, K)
        """
        bna = inputs.size(0)  # batch * n_agents

        # Reshape hidden state: (bna, K, rnn_hidden_dim)
        h_in = hidden_state.reshape(bna, self.n_encoders, self.rnn_hidden_dim)

        # --- Selector forward (trainable, gradient flows here) ---
        alpha = self.selector(inputs)  # (bna, K)

        # --- Encoder forward (frozen, no gradient) ---
        feats = []
        h_outs = []
        with th.no_grad():
            for k, enc in enumerate(self.encoders):
                feat_k, h_k = enc(inputs, h_in[:, k])  # feat_k: (bna, enc_feature_dim)
                feats.append(feat_k)
                h_outs.append(h_k)

        # Stack: (bna, K, enc_feature_dim) and (bna, K, rnn_hidden_dim)
        feats = th.stack(feats, dim=1)    # (bna, K, enc_feature_dim)
        h_out = th.stack(h_outs, dim=1)   # (bna, K, rnn_hidden_dim)

        # --- Weighted combination (detach feats to block encoder gradients) ---
        feats = feats.detach()
        # alpha: (bna, K, 1) * feats: (bna, K, enc_feature_dim) -> sum over K
        combined = (alpha.unsqueeze(-1) * feats).sum(dim=1)  # (bna, enc_feature_dim)

        # --- Q-head (trainable) ---
        q = self.fc2(combined)  # (bna, n_actions)

        # Flatten hidden states back: (bna, K * rnn_hidden_dim)
        h_out = h_out.reshape(bna, -1)

        return q, h_out, alpha

    def load_encoder_from_rnn(self, rnn_state_dict, encoder_idx=0):
        """
        Load a pre-trained RNNAgent's weights into a specific encoder branch.
        Maps: fc1->fc1, rnn->rnn, fc2->fc_enc (if dims match, otherwise skip fc2).
        """
        enc = self.encoders[encoder_idx]
        mapping = {
            "fc1.weight": "fc1.weight",
            "fc1.bias": "fc1.bias",
            "rnn.weight_ih": "rnn.weight_ih",
            "rnn.weight_hh": "rnn.weight_hh",
            "rnn.bias_ih": "rnn.bias_ih",
            "rnn.bias_hh": "rnn.bias_hh",
        }
        enc_sd = enc.state_dict()
        for src_key, dst_key in mapping.items():
            if src_key in rnn_state_dict and dst_key in enc_sd:
                if rnn_state_dict[src_key].shape == enc_sd[dst_key].shape:
                    enc_sd[dst_key] = rnn_state_dict[src_key]
        enc.load_state_dict(enc_sd)
