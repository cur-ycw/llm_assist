"""
SCARE Pipeline Integration Test

Tests the full data flow WITHOUT StarCraft II:
1. Create mock pretrained encoder checkpoints + alpha_prior.json
2. Load them via the run.py SCARE hook logic
3. Build SCARERNNAgent -> BasicMAC -> SCAREQLearner
4. Run one training step on a synthetic EpisodeBatch
5. Verify gradients flow correctly (selector gets grad, encoders don't)

Usage:
    cd D:/ClaudeCodeRepo/smac_test/pymarl
    python src/test_scare_pipeline.py
"""

import os
import sys
import json
import shutil
import tempfile
import logging

# Setup path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import torch as th
import torch.nn as nn
from types import SimpleNamespace as SN

from modules.agents.scare_agent import SCARERNNAgent
from controllers.basic_controller import BasicMAC
from learners.scare_learner import SCAREQLearner
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
# Cannot import scare_pretrain directly (it imports call_llm at module level).
# Copy load_pretrained_encoders inline for testing.
def load_pretrained_encoders(scare_agent, lib_dir):
    """Load pre-trained RNNAgent checkpoints into SCARERNNAgent's encoder bank."""
    encoder_path = os.path.join(lib_dir, 'encoder.json')
    with open(encoder_path, 'r') as f:
        encoder_library = json.load(f)
    for idx_str, enc_info in encoder_library.items():
        idx = int(idx_str) - 1
        model_path = enc_info['model_path']
        agent_path = os.path.join(model_path, 'agent.th')
        if os.path.exists(agent_path):
            rnn_state_dict = th.load(agent_path, map_location='cpu')
            scare_agent.load_encoder_from_rnn(rnn_state_dict, encoder_idx=idx)
            print("  [SCARE] Loaded encoder {} from: {}".format(idx, agent_path))
        else:
            print("  [SCARE] WARNING: Checkpoint not found: {}".format(agent_path))
    scare_agent.freeze_encoders()
    print("  [SCARE] All encoders frozen. Loaded {} encoders.".format(len(encoder_library)))


# ============================================================
# Test configuration
# ============================================================
N_AGENTS = 3
N_ACTIONS = 9
OBS_SHAPE = 30
STATE_SHAPE = 48
N_ENCODERS = 4
RNN_HIDDEN_DIM = 64
ENC_FEATURE_DIM = 64
BS = 4          # batch size
T = 10          # episode length


def make_args():
    """Create a minimal args namespace mimicking PyMARL config."""
    return SN(
        n_agents=N_AGENTS,
        n_actions=N_ACTIONS,
        state_shape=STATE_SHAPE,
        obs_shape=OBS_SHAPE,
        rnn_hidden_dim=RNN_HIDDEN_DIM,
        n_encoders=N_ENCODERS,
        enc_feature_dim=ENC_FEATURE_DIM,
        selector_hidden_dim=64,
        selector_lr=0.0005,
        agent="scare",
        learner="scare_learner",
        agent_output_type="q",
        action_selector="epsilon_greedy",
        epsilon_start=1.0,
        epsilon_finish=0.05,
        epsilon_anneal_time=50000,
        mask_before_softmax=True,
        mixer="qmix",
        mixing_embed_dim=32,
        hypernet_layers=2,
        hypernet_embed=64,
        double_q=True,
        gamma=0.99,
        lr=0.0005,
        optim_alpha=0.99,
        optim_eps=1e-5,
        grad_norm_clip=10,
        target_update_interval=200,
        lambda_prior_start=1.0,
        lambda_prior_finish=0.01,
        lambda_prior_anneal=200000,
        learner_log_interval=1000,
        obs_last_action=False,
        obs_agent_id=False,
        mac="basic_mac",
        use_cuda=False,
        device="cpu",
        scare_lib_dir="",
    )


def make_fake_scare_lib(tmp_dir, input_shape):
    """Create mock pretrained encoder outputs."""
    lib_dir = os.path.join(tmp_dir, "scare_lib", "test_run")
    os.makedirs(lib_dir, exist_ok=True)

    # Create K fake RNNAgent checkpoints
    encoder_library = {}
    for k in range(1, N_ENCODERS + 1):
        model_dir = os.path.join(lib_dir, "models", "encoder_{}".format(k))
        os.makedirs(model_dir, exist_ok=True)

        # Mimic a standard RNNAgent state_dict
        fake_sd = {
            "fc1.weight": th.randn(RNN_HIDDEN_DIM, input_shape),
            "fc1.bias": th.randn(RNN_HIDDEN_DIM),
            "rnn.weight_ih": th.randn(3 * RNN_HIDDEN_DIM, RNN_HIDDEN_DIM),
            "rnn.weight_hh": th.randn(3 * RNN_HIDDEN_DIM, RNN_HIDDEN_DIM),
            "rnn.bias_ih": th.randn(3 * RNN_HIDDEN_DIM),
            "rnn.bias_hh": th.randn(3 * RNN_HIDDEN_DIM),
            "fc2.weight": th.randn(N_ACTIONS, RNN_HIDDEN_DIM),
            "fc2.bias": th.randn(N_ACTIONS),
        }
        th.save(fake_sd, os.path.join(model_dir, "agent.th"))

        encoder_library[str(k)] = {
            "behavior": "test_behavior_{}".format(k),
            "model_path": model_dir,
        }

    # Save encoder.json
    with open(os.path.join(lib_dir, "encoder.json"), "w") as f:
        json.dump(encoder_library, f)

    # Save alpha_prior.json (K x K matrix as nested dict, 1-indexed)
    alpha_prior = {}
    for qi in range(1, N_ENCODERS + 1):
        scores = {}
        for pi in range(1, N_ENCODERS + 1):
            # Give higher weight to matching encoder
            scores[str(pi)] = 2.0 if pi == qi else 0.5
        alpha_prior[str(qi)] = scores

    with open(os.path.join(lib_dir, "alpha_prior.json"), "w") as f:
        json.dump(alpha_prior, f)

    return lib_dir


def make_batch(scheme, groups):
    """Create a synthetic EpisodeBatch with random data."""
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=N_ACTIONS)])
    }
    batch = EpisodeBatch(scheme, groups, BS, T + 1, preprocess=preprocess, device="cpu")

    # Fill with random data
    for t in range(T + 1):
        obs = th.randn(BS, N_AGENTS, OBS_SHAPE)
        state = th.randn(BS, STATE_SHAPE)
        avail = th.ones(BS, N_AGENTS, N_ACTIONS, dtype=th.int)
        actions = th.randint(0, N_ACTIONS, (BS, N_AGENTS, 1))
        reward = th.randn(BS, 1)
        terminated = th.zeros(BS, 1, dtype=th.uint8)
        if t == T:
            terminated = th.ones(BS, 1, dtype=th.uint8)

        data = {
            "obs": obs,
            "state": state,
            "avail_actions": avail,
            "actions": actions,
            "reward": reward,
            "terminated": terminated,
        }
        batch.update(data, ts=slice(t, t + 1))

    # Mark filled
    batch.data.transition_data["filled"][:, :T] = 1

    return batch


class DummyLogger:
    """Minimal logger for testing."""
    def __init__(self):
        self.console_logger = logging.getLogger("test_scare")
        self.console_logger.setLevel(logging.INFO)
        if not self.console_logger.handlers:
            self.console_logger.addHandler(logging.StreamHandler())
        self.stats = {}

    def log_stat(self, key, value, t_env):
        self.stats[key] = value

    def print_recent_stats(self):
        pass


# ============================================================
# Tests
# ============================================================

def test_1_agent_forward():
    """Test SCARERNNAgent forward pass returns correct shapes."""
    print("\n=== Test 1: SCARERNNAgent forward ===")
    args = make_args()
    agent = SCARERNNAgent(OBS_SHAPE, args)

    bna = BS * N_AGENTS
    inputs = th.randn(bna, OBS_SHAPE)
    h = agent.init_hidden().expand(bna, -1)

    q, h_out, alpha = agent(inputs, h)

    assert q.shape == (bna, N_ACTIONS), "q shape: {} != {}".format(q.shape, (bna, N_ACTIONS))
    assert h_out.shape == (bna, N_ENCODERS * RNN_HIDDEN_DIM), "h shape mismatch"
    assert alpha.shape == (bna, N_ENCODERS), "alpha shape mismatch"
    assert th.allclose(alpha.sum(dim=-1), th.ones(bna)), "alpha should sum to 1"

    print("  PASSED: shapes q={}, h={}, alpha={}".format(q.shape, h_out.shape, alpha.shape))


def test_2_encoder_loading():
    """Test load_pretrained_encoders loads weights correctly."""
    print("\n=== Test 2: Encoder loading ===")
    args = make_args()
    agent = SCARERNNAgent(OBS_SHAPE, args)

    tmp_dir = tempfile.mkdtemp()
    try:
        lib_dir = make_fake_scare_lib(tmp_dir, OBS_SHAPE)

        # Record original weights
        orig_w = agent.encoders[0].fc1.weight.clone()

        load_pretrained_encoders(agent, lib_dir)

        # Weights should have changed
        new_w = agent.encoders[0].fc1.weight
        assert not th.equal(orig_w, new_w), "Encoder weights should change after loading"

        # Encoders should be frozen
        for k, enc in enumerate(agent.encoders):
            for p in enc.parameters():
                assert not p.requires_grad, "Encoder {} param should be frozen".format(k)

        print("  PASSED: encoders loaded and frozen")
    finally:
        shutil.rmtree(tmp_dir)


def test_3_alpha_prior_build():
    """Test _build_alpha_prior converts JSON dict to correct tensor."""
    print("\n=== Test 3: Alpha prior build ===")
    args = make_args()
    logger = DummyLogger()

    scheme = {
        "obs": {"vshape": OBS_SHAPE, "group": "agents"},
        "state": {"vshape": STATE_SHAPE},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (N_ACTIONS,), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": N_AGENTS}

    mac = BasicMAC(scheme, groups, args)
    learner = SCAREQLearner(mac, scheme, logger, args)

    # No prior yet
    assert learner.alpha_prior is None, "alpha_prior should be None initially"

    # Build from dict
    prior_dict = {
        "1": {"1": 2.0, "2": 0.5, "3": 0.5, "4": 0.5},
        "2": {"1": 0.5, "2": 2.0, "3": 0.5, "4": 0.5},
        "3": {"1": 0.5, "2": 0.5, "3": 2.0, "4": 0.5},
        "4": {"1": 0.5, "2": 0.5, "3": 0.5, "4": 2.0},
    }
    learner._build_alpha_prior(prior_dict)

    assert learner.alpha_prior is not None, "alpha_prior should be set"
    assert learner.alpha_prior.shape == (N_ENCODERS, N_ENCODERS), \
        "shape: {} != ({}, {})".format(learner.alpha_prior.shape, N_ENCODERS, N_ENCODERS)

    # Rows should sum to 1 (normalized)
    row_sums = learner.alpha_prior.sum(dim=-1)
    assert th.allclose(row_sums, th.ones(N_ENCODERS), atol=1e-5), \
        "Rows should sum to 1, got: {}".format(row_sums)

    # Diagonal should be highest
    for i in range(N_ENCODERS):
        assert learner.alpha_prior[i, i] == learner.alpha_prior[i].max(), \
            "Diagonal should be max in row {}".format(i)

    print("  PASSED: alpha_prior shape={}, rows sum to 1, diagonal dominant".format(
        learner.alpha_prior.shape))


def test_4_run_hook_integration():
    """Test the run.py SCARE init hook logic (without actually calling run_sequential)."""
    print("\n=== Test 4: run.py SCARE hook integration ===")
    args = make_args()
    logger = DummyLogger()

    scheme = {
        "obs": {"vshape": OBS_SHAPE, "group": "agents"},
        "state": {"vshape": STATE_SHAPE},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (N_ACTIONS,), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": N_AGENTS}

    mac = BasicMAC(scheme, groups, args)
    learner = SCAREQLearner(mac, scheme, logger, args)

    tmp_dir = tempfile.mkdtemp()
    try:
        lib_dir = make_fake_scare_lib(tmp_dir, OBS_SHAPE)
        args.scare_lib_dir = lib_dir

        # Simulate the run.py hook
        if args.learner == "scare_learner" and getattr(args, "scare_lib_dir", ""):
            load_pretrained_encoders(mac.agent, args.scare_lib_dir)

            prior_path = os.path.join(args.scare_lib_dir, 'alpha_prior.json')
            if os.path.exists(prior_path):
                with open(prior_path, 'r') as f:
                    args.alpha_prior = json.load(f)
                learner._build_alpha_prior(args.alpha_prior)

        # Verify
        assert learner.alpha_prior is not None, "alpha_prior should be loaded"
        for p in mac.agent.encoders[0].parameters():
            assert not p.requires_grad, "Encoders should be frozen"

        print("  PASSED: hook loaded encoders + alpha_prior successfully")
    finally:
        shutil.rmtree(tmp_dir)


def test_5_training_step():
    """Test one full training step: forward + TD loss + selector loss + backward."""
    print("\n=== Test 5: Full training step ===")
    args = make_args()
    logger = DummyLogger()

    scheme = {
        "obs": {"vshape": OBS_SHAPE, "group": "agents"},
        "state": {"vshape": STATE_SHAPE},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (N_ACTIONS,), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": N_AGENTS}

    mac = BasicMAC(scheme, groups, args)
    learner = SCAREQLearner(mac, scheme, logger, args)

    # Load fake pretrained data
    tmp_dir = tempfile.mkdtemp()
    try:
        lib_dir = make_fake_scare_lib(tmp_dir, OBS_SHAPE)
        load_pretrained_encoders(mac.agent, lib_dir)

        prior_path = os.path.join(lib_dir, 'alpha_prior.json')
        with open(prior_path, 'r') as f:
            prior_dict = json.load(f)
        learner._build_alpha_prior(prior_dict)

        # Create synthetic batch
        batch = make_batch(scheme, groups)

        # Record selector weights before training
        sel_w_before = mac.agent.selector.net[0].weight.clone()
        fc2_w_before = mac.agent.fc2.weight.clone()

        # Encoder weights (should NOT change)
        enc_w_before = mac.agent.encoders[0].fc1.weight.clone()

        # Run one training step
        learner.train(batch, t_env=100, episode_num=1)

        # Verify encoder weights unchanged
        enc_w_after = mac.agent.encoders[0].fc1.weight
        assert th.equal(enc_w_before, enc_w_after), "Encoder weights should NOT change"

        # Verify fc2 weights changed (TD loss)
        fc2_w_after = mac.agent.fc2.weight
        assert not th.equal(fc2_w_before, fc2_w_after), "fc2 weights should change from TD loss"

        # Verify selector weights changed (selector loss)
        sel_w_after = mac.agent.selector.net[0].weight
        assert not th.equal(sel_w_before, sel_w_after), "Selector weights should change"

        # Check logged stats
        assert "td_loss" in logger.stats, "td_loss should be logged"
        assert "selector_loss" in logger.stats, "selector_loss should be logged"
        assert "prior_loss" in logger.stats, "prior_loss should be logged"
        assert "routing_loss" in logger.stats, "routing_loss should be logged"

        print("  PASSED: training step completed")
        print("    td_loss={:.4f}, selector_loss={:.4f}".format(
            logger.stats["td_loss"], logger.stats["selector_loss"]))
        print("    routing_loss={:.4f}, prior_loss={:.4f}".format(
            logger.stats["routing_loss"], logger.stats["prior_loss"]))
        print("    Encoder frozen: OK, fc2 updated: OK, selector updated: OK")
    finally:
        shutil.rmtree(tmp_dir)


def test_6_prior_loss_with_and_without_prior():
    """Test that prior loss differs between uniform fallback and loaded prior."""
    print("\n=== Test 6: Prior loss with vs without alpha_prior ===")
    args = make_args()
    logger = DummyLogger()

    scheme = {
        "obs": {"vshape": OBS_SHAPE, "group": "agents"},
        "state": {"vshape": STATE_SHAPE},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (N_ACTIONS,), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": N_AGENTS}

    # --- Without prior (uniform fallback) ---
    mac1 = BasicMAC(scheme, groups, args)
    learner1 = SCAREQLearner(mac1, scheme, logger, args)
    assert learner1.alpha_prior is None

    # Create a fake alpha_out
    alpha_out = th.softmax(th.randn(BS, T + 1, N_AGENTS, N_ENCODERS), dim=-1)
    mask = th.ones(BS, T, 1)
    batch_mock = SN(scheme=scheme, device="cpu")

    loss_uniform = learner1._compute_prior_loss(alpha_out, batch_mock, mask)

    # --- With prior ---
    mac2 = BasicMAC(scheme, groups, args)
    learner2 = SCAREQLearner(mac2, scheme, logger, args)
    prior_dict = {
        "1": {"1": 5.0, "2": 0.1, "3": 0.1, "4": 0.1},
        "2": {"1": 0.1, "2": 5.0, "3": 0.1, "4": 0.1},
        "3": {"1": 0.1, "2": 0.1, "3": 5.0, "4": 0.1},
        "4": {"1": 0.1, "2": 0.1, "3": 0.1, "4": 5.0},
    }
    learner2._build_alpha_prior(prior_dict)
    assert learner2.alpha_prior is not None

    loss_prior = learner2._compute_prior_loss(alpha_out, batch_mock, mask)

    print("  Uniform prior loss: {:.6f}".format(loss_uniform.item()))
    print("  Loaded prior loss:  {:.6f}".format(loss_prior.item()))
    assert loss_uniform.item() != loss_prior.item(), \
        "Prior loss should differ between uniform and loaded prior"

    print("  PASSED: prior losses differ as expected")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SCARE Pipeline Integration Test")
    print("=" * 60)

    passed = 0
    failed = 0
    tests = [
        test_1_agent_forward,
        test_2_encoder_loading,
        test_3_alpha_prior_build,
        test_4_run_hook_integration,
        test_5_training_step,
        test_6_prior_loss_with_and_without_prior,
    ]

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print("  FAILED: {}".format(e))
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Results: {}/{} passed, {} failed".format(passed, len(tests), failed))
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
