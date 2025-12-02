#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_ablation_studies.py

Ablations & Stress Tests for RLMetaDQN vs. Heuristics (TPDS reproducibility).

This script reuses the simulation environment, reward function, and scheduler
definitions from `run_simulated_experiments_tpds.py` and focuses on:

1) State-feature ablations for RLMetaDQN:
   - baseline        : full state
   - no_intensity    : zero out intensity one-hot (indices 4–6)
   - no_workload     : zero out workload one-hot (indices 0–3)
   - no_queue_feat   : zero out queue-length feature (index 8)
   - no_gpu_feat     : zero out GPU-count feature (index 7)

2) Stress tests:
   - Evaluate under moderate and high intensities across multiple workloads and
     GPU counts (including high-load regimes).

For each (workload, intensity, GPUs, ablation, policy) we collect:

    avg_jct
    p95_jct
    p99_jct
    slowdown_mean
    slowdown_p95
    sla
    gpu_utilization
    energy_kwh
    queue_len_mean
    queue_len_max
    queue_exploded (1 if queue_len_max > 200, else 0)
    jct_cv
    fairness_jain
    action_entropy (0 for fixed heuristics)
    oracle_gap_jct (only for RLMetaDQN rows, vs best heuristic avg_jct)

Output:
    ablation_metrics.csv  (configurable via --output)
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Try to make stdout line-buffered so progress messages appear immediately
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ----------------------------------------------------------------------
# Import core simulation pieces
# ----------------------------------------------------------------------

import run_simulated_experiments_tpds as sim  # type: ignore

SCHEDULER_NAMES = sim.SCHEDULER_NAMES  # ["StaticEqual", "StaticPriority", "EDF", "SRPT", "DynamicHeuristic"]


# ----------------------------------------------------------------------
# Small ASCII progress bar helper (copied from core sim script)
# ----------------------------------------------------------------------

def format_bar(current: int, total: int, width: int = 20) -> str:
    if total <= 0:
        total = 1
    frac = current / total
    filled = int(width * frac)
    empty = width - filled
    return "[" + "#" * filled + "-" * empty + f"] {current}/{total}"


# ----------------------------------------------------------------------
# Ablation configuration
# ----------------------------------------------------------------------

@dataclass
class AblationSpec:
    name: str
    zero_indices: Optional[List[int]] = None


ABLATIONS: List[AblationSpec] = [
    AblationSpec("baseline", zero_indices=None),
    AblationSpec("no_intensity", zero_indices=[4, 5, 6]),
    AblationSpec("no_workload", zero_indices=[0, 1, 2, 3]),
    AblationSpec("no_queue_feat", zero_indices=[8]),
    AblationSpec("no_gpu_feat", zero_indices=[7]),
]


def apply_state_ablation(state: np.ndarray, spec: AblationSpec) -> np.ndarray:
    """
    Apply simple zeroing-based feature ablation on the 9-D state:

    Indices (from run_simulated_experiments_tpds.encode_state):
        0–3 : workload one-hot
        4–6 : intensity one-hot
        7   : normalized num_gpus
        8   : queue-length / load proxy
    """
    if spec.zero_indices is None:
        return state
    s = state.copy()
    for idx in spec.zero_indices:
        if 0 <= idx < s.shape[0]:
            s[idx] = 0.0
    return s


# ----------------------------------------------------------------------
# Seeding utilities
# ----------------------------------------------------------------------

def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# Metrics computation
# ----------------------------------------------------------------------

def compute_metrics_from_env(
    env: sim.SimEnv,
    cfg: sim.SimConfig,
    queue_series: List[int],
    busy_series: List[int],
    action_entropies: List[float],
) -> Dict[str, float]:
    """
    Aggregate metrics for a completed SimEnv run.

    Uses only information available from SimEnv and simple time-series of
    queue length and busy GPUs (collected externally).
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        jcts = []
        slows = []
        sla_flags = []

        for j in env.completed:
            if j.completion_time is None or j.slowdown is None:
                continue
            jcts.append(float(j.completion_time))
            slows.append(float(j.slowdown))
            sla_flags.append(1.0 if not j.sla_violated else 0.0)

        if not jcts:
            raise RuntimeError("No completed jobs in this simulation run.")

        jcts_arr = np.asarray(jcts, dtype=float)
        slows_arr = np.asarray(slows, dtype=float)
        sla_arr = np.asarray(sla_flags, dtype=float)

        avg_jct = float(jcts_arr.mean())
        p95_jct = float(np.percentile(jcts_arr, 95))
        p99_jct = float(np.percentile(jcts_arr, 99))

        slowdown_mean = float(slows_arr.mean())
        slowdown_p95 = float(np.percentile(slows_arr, 95))

        sla = float(sla_arr.mean())

        queue_arr = np.asarray(queue_series, dtype=float) if queue_series else np.array([0.0])
        busy_arr = np.asarray(busy_series, dtype=float) if busy_series else np.array([0.0])
        ent_arr = np.asarray(action_entropies, dtype=float) if action_entropies else np.array([0.0])

        queue_len_mean = float(queue_arr.mean())
        queue_len_max = float(queue_arr.max())
        queue_exploded = 1.0 if queue_len_max > 200.0 else 0.0

        util_arr = busy_arr / max(cfg.num_gpus, 1)
        gpu_utilization = float(util_arr.mean())

        # Energy estimate (same simple model as elsewhere)
        energy_kwh = 0.0
        for busy in busy_series:
            power_w = cfg.power_idle_w + busy * cfg.power_per_gpu_w
            energy_kwh += (power_w * (cfg.dt_s / 3600.0)) / 1000.0
        energy_kwh = float(energy_kwh)

        jct_std = float(jcts_arr.std(ddof=0))
        jct_cv = float(jct_std / (avg_jct + 1e-9))

        # Jain fairness index on normalized JCTs
        x = jcts_arr
        x_mean = x.mean() + 1e-9
        x_norm = x / x_mean
        x_norm = np.clip(x_norm, -1e6, 1e6)

        num = (x_norm.sum() ** 2)
        den = len(x_norm) * (np.square(x_norm).sum() + 1e-9)
        fairness_jain_raw = num / den if den > 0 else 0.0
        fairness_jain = float(min(max(fairness_jain_raw, 0.0), 1.0))

        action_entropy = float(ent_arr.mean())

    return {
        "avg_jct": avg_jct,
        "p95_jct": p95_jct,
        "p99_jct": p99_jct,
        "slowdown_mean": slowdown_mean,
        "slowdown_p95": slowdown_p95,
        "sla": sla,
        "gpu_utilization": gpu_utilization,
        "energy_kwh": energy_kwh,
        "queue_len_mean": queue_len_mean,
        "queue_len_max": queue_len_max,
        "queue_exploded": queue_exploded,
        "jct_cv": jct_cv,
        "fairness_jain": fairness_jain,
        "action_entropy": action_entropy,
    }


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------

def eval_heuristic(
    env_cfg: sim.SimConfig,
    scheduler_name: str,
    seeds: List[int],
) -> Dict[str, float]:
    """
    Evaluate a fixed heuristic scheduler on env_cfg across multiple seeds.
    """
    metrics_all: Dict[str, List[float]] = {}

    for seed in seeds:
        seed_all(seed)
        env = sim.SimEnv(env_cfg, seed)
        state = env.reset_state()
        _ = state  # unused but kept for clarity

        queue_series: List[int] = []
        busy_series: List[int] = []
        entropy_series: List[float] = []

        done = False
        while not done:
            # Fixed policy; distribution is one-hot
            probs = np.zeros(len(SCHEDULER_NAMES), dtype=np.float64)
            try:
                idx = SCHEDULER_NAMES.index(scheduler_name)
            except ValueError:
                raise ValueError(f"Unknown scheduler_name {scheduler_name}")
            probs[idx] = 1.0
            ent = 0.0  # entropy of a delta distribution

            next_state, _reward, done = env.step(scheduler_name)
            _ = next_state  # we don't use the state explicitly here

            busy = sum(1 for g in env.gpus if g.job is not None)
            qlen = len(env.pending)

            queue_series.append(qlen)
            busy_series.append(busy)
            entropy_series.append(ent)

        m = compute_metrics_from_env(env, env_cfg, queue_series, busy_series, entropy_series)

        for k, v in m.items():
            metrics_all.setdefault(k, []).append(v)

    # Aggregate over seeds
    return {k: float(np.mean(v)) for k, v in metrics_all.items()}


def eval_rl_policy(
    env_cfg: sim.SimConfig,
    policy_net: sim.DQNNet,
    device: torch.device,
    ablation: AblationSpec,
    seeds: List[int],
) -> Dict[str, float]:
    """
    Evaluate RLMetaDQN under a given state ablation spec on env_cfg across seeds.
    """
    policy_net.eval()
    metrics_all: Dict[str, List[float]] = {}

    for seed in seeds:
        seed_all(seed)
        env = sim.SimEnv(env_cfg, seed)
        state = env.reset_state()

        queue_series: List[int] = []
        busy_series: List[int] = []
        entropy_series: List[float] = []

        done = False
        while not done:
            s_np = np.asarray(state, dtype=np.float32)
            s_abl = apply_state_ablation(s_np, ablation)

            with torch.no_grad():
                s_t = torch.tensor(s_abl, dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = policy_net(s_t).squeeze(0)
                # Softmax over Q for entropy; temperature = 1.0
                p = torch.softmax(q_vals, dim=0)
                a = int(torch.argmax(p).item())
                probs_np = p.cpu().numpy()

            policy_name = SCHEDULER_NAMES[a]

            next_state, _reward, done = env.step(policy_name)

            busy = sum(1 for g in env.gpus if g.job is not None)
            qlen = len(env.pending)

            queue_series.append(qlen)
            busy_series.append(busy)

            probs_np = np.clip(probs_np, 1e-9, 1.0)
            ent = -float(np.sum(probs_np * np.log(probs_np)))
            entropy_series.append(ent)

            state = next_state

        m = compute_metrics_from_env(env, env_cfg, queue_series, busy_series, entropy_series)

        for k, v in m.items():
            metrics_all.setdefault(k, []).append(v)

    return {k: float(np.mean(v)) for k, v in metrics_all.items()}


# ----------------------------------------------------------------------
# Experiment grid
# ----------------------------------------------------------------------

@dataclass
class EnvConfigKey:
    workload: str
    intensity: str
    n_gpus: int
    stress_tag: str  # e.g., "normal" or "stress_high"


def build_env_cfg(key: EnvConfigKey, lam_base: float) -> sim.SimConfig:
    return sim.SimConfig(
        num_gpus=key.n_gpus,
        workload_label=key.workload,
        intensity_label=key.intensity,
        lam_base=lam_base,
        horizon_s=3600.0,
        dt_s=5.0,
        reward_cfg=sim.RewardConfig(),
    )


def get_env_grid() -> List[EnvConfigKey]:
    """
    Define a moderate but representative grid of regimes plus stress cases.

    - 4 workloads
    - intensities: medium (normal), high (stress)
    - GPU counts: 4, 16
    """
    workloads = ["base", "bursty", "diurnal", "shifted_diurnal"]
    gpu_list = [4, 16]
    keys: List[EnvConfigKey] = []

    for wl in workloads:
        for ng in gpu_list:
            keys.append(EnvConfigKey(wl, "medium", ng, "normal"))
            keys.append(EnvConfigKey(wl, "high", ng, "stress_high"))

    return keys


# ----------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------

@dataclass
class RunConfig:
    rl_ckpt: Path
    device_str: str
    num_seeds: int
    output_path: Path
    overwrite: bool

    def to_json(self) -> str:
        return (
            "{\n"
            f'  "rl_ckpt": "{self.rl_ckpt}",\n'
            f'  "device": "{self.device_str}",\n'
            f'  "num_seeds": {self.num_seeds},\n'
            f'  "output_path": "{self.output_path}",\n'
            f'  "overwrite": {str(self.overwrite).lower()}\n'
            "}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation & stress tests for RLMetaDQN vs heuristics."
    )
    parser.add_argument(
        "--rl-ckpt",
        type=str,
        required=True,
        help="Path to rl_meta_dqn_tpds.pt (trained with run_simulated_experiments_tpds.py).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device for RL (e.g., "cuda" or "cpu").',
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Number of seeds per (env, ablation, policy).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ablation_metrics.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rl_ckpt = Path(args.rl_ckpt).expanduser().resolve()
    if not rl_ckpt.exists():
        raise FileNotFoundError(f"RL checkpoint not found: {rl_ckpt}")

    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file '{output_path}' already exists. Use --overwrite to replace it."
        )

    # Resolve device
    if args.device.lower() == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.", flush=True)
        device = torch.device("cpu")
        device_str = "cpu"
    else:
        device = torch.device(args.device)
        device_str = args.device

    cfg = RunConfig(
        rl_ckpt=rl_ckpt,
        device_str=device_str,
        num_seeds=args.num_seeds,
        output_path=output_path,
        overwrite=args.overwrite,
    )

    print("=============================================")
    print("Ablation & Stress Test Configuration")
    print("=============================================")
    print(cfg.to_json())
    print("=============================================\n", flush=True)

    # Load RLMetaDQN
    state_dim = 9  # matches encode_state in run_simulated_experiments_tpds.py
    n_actions = len(SCHEDULER_NAMES)
    policy_net = sim.DQNNet(state_dim=state_dim, n_actions=n_actions).to(device)
    state_dict = torch.load(str(rl_ckpt), map_location=device)
    policy_net.load_state_dict(state_dict)
    policy_net.eval()
    print(f"[LOAD] Loaded RLMetaDQN from: {rl_ckpt}", flush=True)

    # Build environment grid
    env_keys = get_env_grid()
    n_envs = len(env_keys)

    base_lam = 1.0   # jobs per 60 seconds; same semantic as core script
    lam_base = base_lam / 60.0

    # Seed list for evaluation
    seeds = [42 + i for i in range(cfg.num_seeds)]

    rows: List[Dict[str, float]] = []

    print(f"[INFO] Workloads × Intensities × GPU configs: {n_envs}")
    print(f"[INFO] Ablations: {len(ABLATIONS)}")
    print("[INFO] Starting ablation & stress experiments...\n", flush=True)

    for env_idx, key in enumerate(env_keys, start=1):
        env_cfg = build_env_cfg(key, lam_base)
        print(
            f"[ENV ] {format_bar(env_idx, n_envs)} "
            f"workload={key.workload}, intensity={key.intensity}, "
            f"GPUs={key.n_gpus}, tag={key.stress_tag}",
            flush=True,
        )

        # --------------------------------------------------------------
        # 1) Heuristic baselines for this env
        # --------------------------------------------------------------
        heur_metrics: Dict[str, Dict[str, float]] = {}
        for sched in SCHEDULER_NAMES:
            print(f"   [HEUR] scheduler={sched} ...", flush=True)
            start = time.time()
            m = eval_heuristic(env_cfg, sched, seeds)
            end = time.time()
            print(
                f"          done in {end - start:.2f}s, avg_jct={m['avg_jct']:.2f}, "
                f"sla={m['sla']:.3f}",
                flush=True,
            )
            heur_metrics[sched] = m

            row = {
                "workload": key.workload,
                "intensity": key.intensity,
                "n_gpus": key.n_gpus,
                "stress_tag": key.stress_tag,
                "policy": sched,
                "policy_type": "heuristic",
                "ablation": "none",
                "oracle_gap_jct": 0.0,  # not applicable
            }
            row.update(m)
            rows.append(row)

        # best heuristic avg_jct for oracle gap
        best_heur_jct = min(m["avg_jct"] for m in heur_metrics.values())

        # --------------------------------------------------------------
        # 2) RLMetaDQN under each ablation spec
        # --------------------------------------------------------------
        for abl_idx, abl in enumerate(ABLATIONS, start=1):
            print(
                f"   [RL  ] {format_bar(abl_idx, len(ABLATIONS), width=15)} "
                f"ablation={abl.name}",
                flush=True,
            )
            start = time.time()
            m_rl = eval_rl_policy(env_cfg, policy_net, device, abl, seeds)
            end = time.time()

            gap = (m_rl["avg_jct"] / best_heur_jct) - 1.0 if best_heur_jct > 0 else 0.0

            print(
                f"          done in {end - start:.2f}s, avg_jct={m_rl['avg_jct']:.2f}, "
                f"sla={m_rl['sla']:.3f}, gap_vs_best_heur={gap * 100:.1f}%",
                flush=True,
            )

            row = {
                "workload": key.workload,
                "intensity": key.intensity,
                "n_gpus": key.n_gpus,
                "stress_tag": key.stress_tag,
                "policy": "RLMetaDQN",
                "policy_type": "rl",
                "ablation": abl.name,
                "oracle_gap_jct": gap,
            }
            row.update(m_rl)
            rows.append(row)

    df = pd.DataFrame(rows)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_path, index=False)

    print("\n=============================================")
    print("Ablation & Stress Tests Completed.")
    print(f"Saved metrics to: {cfg.output_path}")
    print(f"Total rows: {len(df)}")
    print("=============================================", flush=True)


if __name__ == "__main__":
    main()

