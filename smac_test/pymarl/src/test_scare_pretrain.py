"""
SCARE Encoder Pre-training Workflow Test
分步测试 LLM 生成语义 -> 奖励函数 -> 训练 encoder 的完整流程

Usage:
    conda activate pymarl
    cd D:/ClaudeCodeRepo/smac_test/pymarl
    python src/test_scare_pretrain.py
"""

import os
import sys
import re
import json
import time
import textwrap
import datetime

# 确保 src 在 path 中
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# 添加 llm_assist 的 language 目录到 path，以便导入 call_llm
_llm_lang_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(_src_dir))),
    'llm_assist', 'llm_assist-main', 'human_aicoord', 'SemDiv', 'language'
)
if os.path.exists(_llm_lang_dir):
    sys.path.insert(0, _llm_lang_dir)

# ============================================================
# Prompt templates (from SemDiv prompt_text_sc2.py)
# ============================================================
PROMPT_ENV = (
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

PROMPT_CODE = '''
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

PROMPT_MULTI = (
    "\nHuman player teams may have specific cooperation preferences to play "
    "the game, like attacking different enemies. They have their own "
    "`additional_reward` shown in the code.\n"
)

PROMPT_WRITE_CODE = '''
According to this cooperation preference, write an operational and executable reward function that formats as 'def additional_reward(self) -> float' and returns the 'reward : float' only.

1. Please think step by step and tell us what this code means;
2. The code function must align with the cooperation preference.
3. It can be a dense reward that guides the team to learn the cooperation preference.
4. Short and simple code is better.
'''


def print_step(step_num, title):
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {title}")
    print(f"{'='*60}\n")


def print_result(success, msg):
    tag = "[PASS]" if success else "[FAIL]"
    print(f"  {tag} {msg}")
    return success


# ============================================================
# Step 1: Test LLM connection and behavior generation
# ============================================================
def test_step1_llm_generate_behavior():
    print_step(1, "LLM 生成 cooperation preference")

    try:
        from call_llm import LLM
        llm = LLM(mode='openai')
        print_result(True, "LLM 模块导入成功")
    except Exception as e:
        return print_result(False, f"LLM 模块导入失败: {e}")

    prompt = (
        PROMPT_ENV + PROMPT_CODE + PROMPT_MULTI +
        "\nBased on the information above, think step by step to come up with "
        "a possible cooperation preference. The preference should be "
        "deterministic and concrete. It should be as simple as possible. "
        "Avoid conditional terms like if, unless, when, etc. "
        "It should not conflict with the original task objective.\n\n"
        "Finally, output the preference in the format: "
        "'Human players may prefer to {preference}'"
    )

    try:
        print("  Calling LLM to generate behavior...")
        output = llm.call_llm(prompt, big_model=True)

        if output == 'No answer!' or not output:
            return print_result(False, f"LLM 返回空结果: {output}")

        print(f"  LLM raw output (first 300 chars):\n  {output[:300]}...\n")

        # Extract clean behavior
        behavior = llm.call_llm(
            f'Task: Find and extract the part describing the specific cooperation '
            f'preference from the original text, ignoring other analysis related '
            f'content. Output the cooperation preference only.\n\n'
            f'Input: ({output})\n\nOutput: ',
            big_model=False
        )

        if not behavior or behavior == 'No answer!':
            return print_result(False, "无法提取 behavior")

        print(f"  Extracted behavior: {behavior}")
        print_result(True, "Behavior 生成成功")
        return behavior
    except Exception as e:
        return print_result(False, f"LLM 调用异常: {e}")


# ============================================================
# Step 2: Test LLM writing reward function code
# ============================================================
def test_step2_llm_write_reward(behavior):
    print_step(2, "LLM 生成 reward function 代码")

    try:
        from call_llm import LLM
        llm = LLM(mode='openai')
    except Exception as e:
        return print_result(False, f"LLM 导入失败: {e}")

    prompt = (
        PROMPT_ENV + PROMPT_CODE +
        f"\nNow we want to train a team with this specific cooperation behavior:"
        f"\n---\n{behavior}\n---" + PROMPT_WRITE_CODE
    )

    try:
        print("  Calling LLM to write reward function...")
        code_output = llm.call_llm(prompt, big_model=True)

        if not code_output or 'No answer!' in code_output:
            return print_result(False, "LLM 返回空结果")

        if "def additional_reward(self" not in code_output:
            print(f"  LLM output (first 500 chars): {code_output[:500]}")
            return print_result(False, "LLM 输出中没有 additional_reward 函数")

        print_result(True, "LLM 生成了包含 additional_reward 的代码")

        # Extract clean code
        code = llm.call_llm(
            code_output +
            '\n\nTODO: Extract the python function `additional_reward` in the text. '
            'Output the function only (start with "```python\\ndef", end with "\\n```") '
            'so that i can directly copy it into my code.',
            big_model=False
        )
        store_code = code.split("```python\n")[1].split('```')[0]
        print(f"  Extracted code:\n")
        for line in store_code.strip().split('\n'):
            print(f"    {line}")
        print()
        print_result(True, "代码提取成功")
        return store_code
    except Exception as e:
        return print_result(False, f"LLM 调用异常: {e}")


# ============================================================
# Step 3: Test code syntax validity
# ============================================================
def test_step3_code_syntax(code):
    print_step(3, "验证生成代码的语法正确性")

    if not code:
        return print_result(False, "没有代码可验证")

    try:
        compile(code, '<llm_reward>', 'exec')
        print_result(True, "代码语法正确，compile 通过")
        return True
    except SyntaxError as e:
        return print_result(False, f"语法错误: {e}")


# ============================================================
# Step 4: Test standard QMIX training (short run)
# ============================================================
def test_step4_qmix_training():
    print_step(4, "标准 QMIX 短训练测试 (t_max=5000)")

    pymarl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(pymarl_dir, 'src')

    python_exe = sys.executable  # Use the same python that's running this script
    main_py = os.path.join(src_dir, 'main.py')

    cmd = (
        f'cd "{pymarl_dir}" && '
        f'"{python_exe}" "{main_py}" '
        f'--config=qmix --env-config=sc2_v2 '
        f'with t_max=5000 '
        f'test_interval=2000 '
        f'test_nepisode=2 '
        f'save_model=True '
        f'save_model_interval=2000 '
        f'name=scare_encoder_test '
        f'batch_size=4 '
        f'buffer_size=100 '
    )

    print(f"  Running: {cmd[:120]}...")
    print("  (This may take a few minutes...)\n")

    import subprocess
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=600
    )

    if result.returncode != 0:
        print(f"  STDERR (last 500 chars): {result.stderr[-500:]}")
        # 即使 returncode != 0，如果有模型保存也算部分成功
        # PyMARL 的 os._exit 可能导致非零返回码

    # Check if model was saved
    model_dir = os.path.join(pymarl_dir, 'results', 'models')
    model_path = None
    if os.path.exists(model_dir):
        for d in sorted(os.listdir(model_dir), reverse=True):
            if 'scare_encoder_test' in d:
                full_path = os.path.join(model_dir, d)
                # Find the latest timestep subdirectory
                timesteps = [
                    x for x in os.listdir(full_path)
                    if os.path.isdir(os.path.join(full_path, x)) and x.isdigit()
                ]
                if timesteps:
                    latest = max(timesteps, key=int)
                    model_path = os.path.join(full_path, latest)
                    break

    if model_path and os.path.exists(os.path.join(model_path, 'agent.th')):
        print_result(True, f"模型已保存: {model_path}")
        # List saved files
        for f in os.listdir(model_path):
            fsize = os.path.getsize(os.path.join(model_path, f))
            print(f"    {f} ({fsize} bytes)")
        return model_path
    else:
        print(f"  Model dir contents: {os.listdir(model_dir) if os.path.exists(model_dir) else 'NOT FOUND'}")
        return print_result(False, "未找到保存的模型")


# ============================================================
# Step 5: Test loading checkpoint into SCARE encoder bank
# ============================================================
def test_step5_load_into_scare(model_path):
    print_step(5, "加载 checkpoint 到 SCARE encoder bank")

    if not model_path:
        return print_result(False, "没有模型路径")

    import torch as th
    from types import SimpleNamespace

    agent_path = os.path.join(model_path, 'agent.th')
    rnn_state_dict = th.load(agent_path, map_location='cpu')

    print(f"  RNNAgent state_dict keys:")
    for k, v in rnn_state_dict.items():
        print(f"    {k}: {v.shape}")

    # Create a mock args to instantiate SCARERNNAgent
    args = SimpleNamespace(
        n_agents=2,
        n_actions=rnn_state_dict['fc2.weight'].shape[0],
        rnn_hidden_dim=rnn_state_dict['fc1.weight'].shape[0],
        n_encoders=4,
        enc_feature_dim=rnn_state_dict['fc1.weight'].shape[0],  # match rnn_hidden_dim
        selector_hidden_dim=64,
    )
    input_shape = rnn_state_dict['fc1.weight'].shape[1]

    print(f"\n  Inferred: input_shape={input_shape}, rnn_hidden_dim={args.rnn_hidden_dim}, "
          f"n_actions={args.n_actions}")

    try:
        from modules.agents.scare_agent import SCARERNNAgent
        scare_agent = SCARERNNAgent(input_shape, args)
        print_result(True, "SCARERNNAgent 实例化成功")
    except Exception as e:
        return print_result(False, f"SCARERNNAgent 实例化失败: {e}")

    # Load into encoder 0
    try:
        scare_agent.load_encoder_from_rnn(rnn_state_dict, encoder_idx=0)
        print_result(True, "Encoder 0 权重加载成功")
    except Exception as e:
        return print_result(False, f"Encoder 0 加载失败: {e}")

    # Verify weights match
    enc0 = scare_agent.encoders[0]
    match_fc1 = th.allclose(enc0.fc1.weight, rnn_state_dict['fc1.weight'])
    match_rnn = th.allclose(enc0.rnn.weight_ih, rnn_state_dict['rnn.weight_ih'])
    print_result(match_fc1, f"fc1 权重匹配: {match_fc1}")
    print_result(match_rnn, f"GRU weight_ih 匹配: {match_rnn}")

    # Freeze and verify
    scare_agent.freeze_encoders()
    all_frozen = all(
        not p.requires_grad for enc in scare_agent.encoders for p in enc.parameters()
    )
    print_result(all_frozen, f"所有 encoder 已冻结: {all_frozen}")

    # Test forward pass
    try:
        bs = 2 * args.n_agents  # batch * n_agents
        dummy_input = th.randn(bs, input_shape)
        dummy_hidden = scare_agent.init_hidden().expand(bs, -1).contiguous()

        q, h, alpha = scare_agent(dummy_input, dummy_hidden)
        print(f"\n  Forward pass output shapes:")
        print(f"    q: {q.shape} (expected: [{bs}, {args.n_actions}])")
        print(f"    h: {h.shape} (expected: [{bs}, {args.n_encoders * args.rnn_hidden_dim}])")
        print(f"    alpha: {alpha.shape} (expected: [{bs}, {args.n_encoders}])")
        print(f"    alpha[0]: {alpha[0].detach().numpy()} (should sum to 1.0, sum={alpha[0].sum().item():.4f})")

        assert q.shape == (bs, args.n_actions)
        assert alpha.shape == (bs, args.n_encoders)
        assert abs(alpha[0].sum().item() - 1.0) < 1e-5
        print_result(True, "Forward pass 正确")
        return True
    except Exception as e:
        return print_result(False, f"Forward pass 失败: {e}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  SCARE Encoder Pre-training Workflow Test")
    print("=" * 60)

    results = {}

    # Step 1: LLM generate behavior
    behavior = test_step1_llm_generate_behavior()
    results['step1_behavior'] = bool(behavior and behavior is not True)

    if not results['step1_behavior']:
        # Fallback behavior for testing remaining steps
        behavior = "Human players may prefer to kill enemy A together."
        print(f"\n  [FALLBACK] Using default behavior: {behavior}")

    # Step 2: LLM write reward code
    code = test_step2_llm_write_reward(behavior)
    results['step2_code'] = bool(code and code is not True)

    if not results['step2_code']:
        # Fallback code
        code = textwrap.dedent("""\
        def additional_reward(self) -> float:
            reward = 0.0
            d1 = self.agent_enemy_distance("1", "A")
            d2 = self.agent_enemy_distance("2", "A")
            reward -= (d1 + d2) * 0.1
            if self.killed_enemy == "A":
                reward += 5.0
            return reward
        """)
        print(f"\n  [FALLBACK] Using default reward code")

    # Step 3: Syntax check
    results['step3_syntax'] = test_step3_code_syntax(code)

    # Step 4: QMIX short training
    model_path = test_step4_qmix_training()
    results['step4_training'] = bool(model_path and model_path is not True)

    # Step 5: Load into SCARE
    if results['step4_training']:
        results['step5_scare_load'] = test_step5_load_into_scare(model_path)
    else:
        print_step(5, "加载 checkpoint 到 SCARE encoder bank")
        results['step5_scare_load'] = print_result(False, "跳过 - Step 4 未产出模型")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for step, passed in results.items():
        tag = "[PASS]" if passed else "[FAIL]"
        print(f"  {tag} {step}")

    all_pass = all(results.values())
    print(f"\n  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"{'='*60}\n")
