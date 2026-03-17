"""
SCARE End-to-End Training Script

Step 1: Pre-train K encoders via LLM-driven semantic diversity (scare_pretrain.py)
Step 2: Main SCARE training via PyMARL (main.py --config=scare)

Usage:
    # Full pipeline (pretrain + main train):
    python src/scare_train.py --map_name 3m --n_encoders 4 --cuda_id 0

    # Skip pretrain (use existing scare_lib):
    python src/scare_train.py --map_name 3m --scare_lib_dir scare_lib/xxx --skip_pretrain

    # Pretrain only (no main training):
    python src/scare_train.py --map_name 3m --n_encoders 4 --pretrain_only
"""

import os
import sys
import argparse
# Skip -dataVersion when launching SC2 to avoid NGDP:E_NOT_AVAILABLE in offline environments
import subprocess

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PYMARL_ROOT = os.path.dirname(_THIS_DIR)


def run_pretrain(args):
    """Run SCARE encoder pre-training, return lib_dir."""
    # Import here to avoid top-level dependency on call_llm etc.
    sys.path.insert(0, _THIS_DIR)
    from scare_pretrain import SCAREPretrainer

    pretrainer = SCAREPretrainer(
        env=args.env,
        env_file=args.env_file,
        n_encoders=args.n_encoders,
        cuda_id=args.cuda_id,
        map_name=args.map_name,
    )
    pretrainer.run()
    return pretrainer.lib_dir


def run_main_train(args, scare_lib_dir):
    """Launch PyMARL main training with SCARE config."""
    cmd = [
        sys.executable, os.path.join(_THIS_DIR, 'main.py'),
        '--config=scare',
        '--env-config=sc2_scare',
        'with',
        'env_args.map_name={}'.format(args.map_name),
        'scare_lib_dir={}'.format(scare_lib_dir),
        'n_encoders={}'.format(args.n_encoders),
    ]

    if args.cuda_id >= 0:
        cmd.append('use_cuda=True')
    else:
        cmd.append('use_cuda=False')

    if args.t_max:
        cmd.append('t_max={}'.format(args.t_max))

    print("[SCARE] Launching main training:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, cwd=_PYMARL_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description='SCARE End-to-End Training')
    parser.add_argument('--map_name', type=str, default='3m',
                        help='SMAC map name (e.g. 3m, 3s5z, MMM2)')
    parser.add_argument('--n_encoders', type=int, default=4,
                        help='Number of encoders to pre-train')
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='CUDA device ID (-1 for CPU)')
    parser.add_argument('--env', type=str, default='sc2_v1',
                        help='Environment name for pretraining')
    parser.add_argument('--env_file', type=str, default=None,
                        help='Path to env file for reward injection')
    parser.add_argument('--t_max', type=int, default=None,
                        help='Max training timesteps for main training')

    # Pipeline control
    parser.add_argument('--skip_pretrain', action='store_true',
                        help='Skip pretraining, use existing scare_lib_dir')
    parser.add_argument('--pretrain_only', action='store_true',
                        help='Only run pretraining, skip main training')
    parser.add_argument('--scare_lib_dir', type=str, default='',
                        help='Path to existing pretrain output (required if --skip_pretrain)')

    args = parser.parse_args()

    # --- Step 1: Pre-train ---
    if args.skip_pretrain:
        if not args.scare_lib_dir:
            parser.error("--scare_lib_dir is required when using --skip_pretrain")
        scare_lib_dir = args.scare_lib_dir
        print("[SCARE] Skipping pretrain, using: {}".format(scare_lib_dir))
    else:
        print("[SCARE] === Step 1: Encoder Pre-training ===")
        scare_lib_dir = run_pretrain(args)
        print("[SCARE] Pretrain complete. lib_dir: {}".format(scare_lib_dir))

    if args.pretrain_only:
        print("[SCARE] Pretrain-only mode, exiting.")
        return

    # --- Step 2: Main training ---
    print("[SCARE] === Step 2: Main SCARE Training ===")
    run_main_train(args, scare_lib_dir)
    print("[SCARE] Training complete.")


if __name__ == '__main__':
    main()
