"""Feasibility simulation for the SMC annealing schedule (no Isaac Gym, no LLM).

Settles the two hyperparameter risks flagged in the migration review before any RL
compute is spent (plan §6.2, §6.3; review findings #1 and #2):

  #1  Reachability — does ``lambda`` climb from 0 to 1 within ``max_smc_iterations``
      for a given ``(kappa, beta_target, N)``? If the ESS floor forces near-zero
      steps, ``annealing_complete`` (and the Phase-2 acceptance criterion) is never
      reached.

  #2  Collapse — how many distinct lineages survive a chain of systematic
      resamples at small ``N``?

It sweeps ``N x beta_target x kappa`` over several synthetic reward regimes, running
the real ``find_next_lambda`` / ``systematic_resample`` code on each, and prints a
worst-case-over-regimes summary table plus a recommended default cell.

Run:  ``python -m eureka.smc.simulate_schedule``   (from the ``Eureka_smc`` root)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from eureka.smc.resampling import systematic_resample
from eureka.smc.temperature import ess_from_log_weights, find_next_lambda, log_incremental_weights


# --------------------------------------------------------------------------- regimes

def _regime(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic reward vectors spanning easy-to-collapse to well-spread cases."""
    if name == "near_equal":
        return 0.5 + rng.normal(0.0, 0.01, n)
    if name == "one_dominant":
        r = np.full(n, 0.1)
        r[0] = 0.9
        return r
    if name == "bimodal":
        r = np.where(np.arange(n) < n // 2, 0.2, 0.8).astype(float)
        return r
    if name == "beta_realistic":
        return rng.beta(2.0, 5.0, n)  # mean ~0.29, right-skewed like early scores
    if name == "neg_shift":
        return -5.0 + rng.normal(0.0, 0.05, n)  # tests negative rewards
    raise ValueError(name)


REGIMES = ("near_equal", "one_dominant", "bimodal", "beta_realistic", "neg_shift")


# --------------------------------------------------------------------------- one run

@dataclass
class RunResult:
    reached: bool
    n_stages: int
    min_ess: float
    unique_lineages: int


def simulate_cell(
    n: int,
    beta_target: float,
    kappa: float,
    regime: str,
    *,
    min_iters: int,
    max_iters: int,
    jitter: float,
    seed: int,
) -> RunResult:
    """Anneal a single population to lambda=1, tracking ESS and lineage survival.

    Each stage: reset weights to uniform, pick the largest ESS-safe step on the
    current scores, resample, then perturb scores by ``jitter`` to model the MH
    proposal moving each clone (without which collapsed duplicates would look
    identical and stall the schedule).
    """
    rng = np.random.default_rng(seed)
    scores = _regime(regime, n, rng)
    lineages = np.arange(n)
    max_delta = 1.0 / min_iters

    lam = 0.0
    min_ess = float(n)
    stages = 0
    while lam < 1.0 - 1e-9 and stages < max_iters:
        nxt = find_next_lambda(scores, lam, beta_target, kappa, max_delta)
        delta = nxt - lam
        if delta <= 0.0:
            break
        logw = log_incremental_weights(scores, delta, beta_target)
        min_ess = min(min_ess, ess_from_log_weights(logw))

        w = np.exp(logw - logw.max())
        idx = systematic_resample(w, rng)
        scores = scores[idx] + rng.normal(0.0, jitter, n)
        lineages = lineages[idx]

        lam = nxt
        stages += 1

    return RunResult(
        reached=lam >= 1.0 - 1e-9,
        n_stages=stages,
        min_ess=min_ess,
        unique_lineages=int(np.unique(lineages).size),
    )


# --------------------------------------------------------------------------- sweep

@dataclass
class CellSummary:
    n: int
    beta_target: float
    kappa: float
    reached_all: bool
    max_stages: int
    min_ess: float
    min_unique: int  # worst-case surviving lineages across regimes

    @property
    def ok(self) -> bool:
        # Healthy: reaches terminal bridge in every regime AND never collapses to
        # a single lineage.
        return self.reached_all and self.min_unique >= 2


def summarize_cell(n, beta_target, kappa, *, min_iters, max_iters, jitter) -> CellSummary:
    reached_all = True
    max_stages = 0
    min_ess = float("inf")
    min_unique = n
    for r_i, regime in enumerate(REGIMES):
        res = simulate_cell(
            n, beta_target, kappa, regime,
            min_iters=min_iters, max_iters=max_iters, jitter=jitter, seed=1000 + r_i,
        )
        reached_all &= res.reached
        max_stages = max(max_stages, res.n_stages)
        min_ess = min(min_ess, res.min_ess)
        min_unique = min(min_unique, res.unique_lineages)
    return CellSummary(n, beta_target, kappa, reached_all, max_stages, min_ess, min_unique)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--beta", type=float, nargs="+", default=[2.0, 5.0, 10.0, 20.0])
    p.add_argument("--kappa", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    p.add_argument("--min-iters", type=int, default=3, help="max_delta = 1/min_iters")
    p.add_argument("--max-iters", type=int, default=15, help="max_smc_iterations budget")
    p.add_argument("--jitter", type=float, default=0.05, help="MH score perturbation std")
    args = p.parse_args()

    print(f"# SMC schedule feasibility  (min_iters={args.min_iters} -> max_delta="
          f"{1/args.min_iters:.3f}, max_iters={args.max_iters}, jitter={args.jitter})")
    print(f"# regimes: {', '.join(REGIMES)}  | worst-case over regimes per cell\n")
    header = f"{'N':>3} {'beta':>5} {'kappa':>5} | {'reachedAll':>10} {'maxStg':>6} " \
             f"{'minESS':>7} {'minUniq':>7}  flag"
    print(header)
    print("-" * len(header))

    cells: list[CellSummary] = []
    for n in args.n:
        for beta in args.beta:
            for kappa in args.kappa:
                c = summarize_cell(n, beta, kappa,
                                   min_iters=args.min_iters, max_iters=args.max_iters,
                                   jitter=args.jitter)
                cells.append(c)
                flag = "" if c.ok else ("NOT-REACHED" if not c.reached_all else "COLLAPSE")
                print(f"{c.n:>3} {c.beta_target:>5.0f} {c.kappa:>5.2f} | "
                      f"{str(c.reached_all):>10} {c.max_stages:>6} {c.min_ess:>7.2f} "
                      f"{c.min_unique:>7}  {flag}")
        print()

    # Recommendation: among healthy cells, prefer more surviving lineages, then a
    # moderate stage count (well inside budget), then smaller beta (more
    # interpretable acceptance), then smaller N (cheaper — each particle is one RL
    # train). Restrict to N <= 8 for Cartpole-scale cost.
    healthy = [c for c in cells if c.ok and c.n <= 8]
    print("=" * len(header))
    if not healthy:
        print("NO healthy (kappa,beta,N) with N<=8 — widen the sweep or relax min_iters.")
        return
    best = max(healthy, key=lambda c: (c.min_unique, -c.max_stages, -c.beta_target, -c.n))
    print(f"RECOMMENDED (Cartpole-scale, N<=8): "
          f"N={best.n}, beta_target={best.beta_target:.0f}, kappa={best.kappa:.2f}  "
          f"(reaches lambda=1 in <={best.max_stages} stages, "
          f">= {best.min_unique} lineages survive worst-case)")
    plan_default = next((c for c in cells if c.n == 8 and c.beta_target == 5 and c.kappa == 0.9), None)
    if plan_default is not None and not plan_default.ok:
        why = "never reaches lambda=1" if not plan_default.reached_all else "collapses to 1 lineage"
        print(f"NOTE: the plan's config default (N=8, beta=5, kappa=0.9) is degenerate "
              f"here — {why}.")


if __name__ == "__main__":
    main()
