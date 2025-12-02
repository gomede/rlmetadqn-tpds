#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_core_sim.py

Core simulation experiments for TPDS reproducibility.

Uses the simulation environment, reward, and job model defined in
run_simulated_experiments_tpds.py, but focuses purely on evaluation:

- Regimes:
    * 4 workload shapes: base, bursty, diurnal, shifted_diurnal
    * 3 intensities: low, medium, high
    * 4 GPU counts: 2, 4, 8, 16
    * Super-high-load regime (150% intensity) on 4 & 16 GPUs

- Schedulers:
    * StaticEqual
    * StaticPriority
    * EDF
    * SRPT
    * DynamicHeuristic
    * RLMetaDQN (meta-scheduler over the 5 heuristics)

- Metrics per (regime, scheduler, seed):
    * avg_jct
    * p95_jct
    * slowdown_mean
    * slowdown_p95
    * sla                 (fraction of jobs meeting SLA)
    * gpu_utilization     (0–1 average over time)
    * energy_kwh
    * queue_len_mean
    * jct_cv              (coefficient of variation of JCT)
    * fairness_jain       (Jain index over JCT)
    * scheduler_overhead_ms
    * action_entropy      (mean policy entropy; 0 for heuristics)

Output:
    core_sim_metrics.csv  (configurable)
"""

import argparse
import itertools
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import run_simulated_experiments_tpds as sim

# ----------------------------------------------------------------------
# Import core pieces from your existing simulation script
# ----------------------------------------------------------------------

SCHEDULER_NAMES = sim.SCHEDULER_NAMES          # ["StaticEqual", "StaticPriority", "EDF", "SRPT", "DynamicHeuristic"]
SimConfig = sim.SimConfig
SimEnv = sim.SimEnv
RewardConfig = sim.RewardConfig
encode_state = sim.encode_state
DQNNet = sim.DQNNet

# ----------------------------------------------------------------------
# Regime space
# ----------------------------------------------------------------------

WORKLOAD_SHAPES = ["base", "bursty", "diurnal", "shifted_diurnal"]
INTENSITIES = ["low", "medium", "high"]
GPU_COUNTS = [2, 4, 8, 16]

# Super-high-load regime label (150% intensity on 4 & 16 GPUs)
SUPER_HIGH_INTENSITY_LABEL = "super_high_150"
SUPER_HIGH_GPU_COUNTS = [4, 16]

# Arrival rate: same base config as run_simulated_experiments_tpds.py
BASE_LAM = 1.0      # jobs per 60s
LAM_BASE = BASE_LAM / 60.0  # per second

# Cache for RLMetaDQN policies so we don't reload the checkpoint every run
_RL_POLICY_CACHE: Dict[Tuple[str, str], torch.nn.Module] = {}


@dataclass(frozen=True)
class RegimeConfig:
    workload: str
    intensity: str
    num_gpus: int
    super_high_load: bool = False

    def regime_id(self) -> str:
        if self.super_high_load:
            return f"{self.workload}_{SUPER_HIGH_INTENSITY_LABEL}_{self.num_gpus}gpus"
        return f"{self.workload}_{self.intensity}_{self.num_gpus}gpus"


@dataclass
class RunConfig:
    num_seeds: int
    schedulers: List[str]
    rl_ckpt_path: Optional[Path]
    device: str
    output_path: Path
    overwrite: bool
    log_actions: bool

    def to_json(self) -> str:
        return json.dumps(
            {
                "num_seeds": self.num_seeds,
                "schedulers": self.schedulers,
                "rl_ckpt_path": str(self.rl_ckpt_path) if self.rl_ckpt_path else None,
                "device": self.device,
                "output_path": str(self.output_path),
                "overwrite": self.overwrite,
                "log_actions": self.log_actions,
            },
            indent=2,
        )


# ----------------------------------------------------------------------
# Regime enumeration
# ----------------------------------------------------------------------

def enumerate_core_regimes() -> List[RegimeConfig]:
    regimes: List[RegimeConfig] = []

    # 48 regimes: 4 workloads × 3 intensities × 4 GPU counts
    for workload, intensity, num_gpus in itertools.product(
        WORKLOAD_SHAPES, INTENSITIES, GPU_COUNTS
    ):
        regimes.append(
            RegimeConfig(
                workload=workload,
                intensity=intensity,
                num_gpus=num_gpus,
                super_high_load=False,
            )
        )

    # Super-high-load: 150% arrival rate on 4 & 16 GPUs
    for workload, num_gpus in itertools.product(WORKLOAD_SHAPES, SUPER_HIGH_GPU_COUNTS):
        regimes.append(
            RegimeConfig(
                workload=workload,
                intensity=SUPER_HIGH_INTENSITY_LABEL,
                num_gpus=num_gpus,
                super_high_load=True,
            )
        )

    return regimes


# ----------------------------------------------------------------------
# Reproducibility helpers
# ----------------------------------------------------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# RL policy loading (RLMetaDQN)
# ----------------------------------------------------------------------

def get_rl_policy(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    key = (str(ckpt_path), str(device))
    if key in _RL_POLICY_CACHE:
        return _RL_POLICY_CACHE[key]

    state_dim = 9
    n_actions = len(SCHEDULER_NAMES)
    net = DQNNet(state_dim=state_dim, n_actions=n_actions).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    net.load_state_dict(state)
    net.eval()

    _RL_POLICY_CACHE[key] = net
    return net


# ----------------------------------------------------------------------
# Single-run simulator (one regime, one scheduler, one seed)
# ----------------------------------------------------------------------

def simulate_regime(
    workload: str,
    intensity: str,
    num_gpus: int,
    scheduler_name: str,
    seed: int,
    super_high_load: bool = False,
    rl_ckpt_path: Optional[Path] = None,
    device: str = "cpu",
    log_actions: bool = False,
) -> Dict[str, float]:
    """
    Run ONE full simulation and return metrics.
    """

    # Map "super_high_150" to intensity="high" + 1.5× arrival rate
    if intensity == SUPER_HIGH_INTENSITY_LABEL:
        env_intensity = "high"
        lam_scale = 1.5
    else:
        env_intensity = intensity
        lam_scale = 1.0

    lam_base_eff = LAM_BASE * lam_scale

    env_cfg = SimConfig(
        num_gpus=num_gpus,
        workload_label=workload,
        intensity_label=env_intensity,
        lam_base=lam_base_eff,
        horizon_s=3600.0,
        dt_s=5.0,
        reward_cfg=RewardConfig(),
    )

    # Build env
    env = SimEnv(env_cfg, seed)

    # RL policy (if needed)
    rl_net: Optional[torch.nn.Module] = None
    torch_device = torch.device(device)

    if scheduler_name == "RLMetaDQN":
        if rl_ckpt_path is None:
            raise ValueError("RLMetaDQN requires --rl-ckpt path")
        rl_net = get_rl_policy(rl_ckpt_path, torch_device)

    # Buffers for metrics
    jcts: List[float] = []
    slows: List[float] = []
    sla_flags: List[float] = []
    queue_lengths: List[float] = []
    gpu_utils: List[float] = []
    energy_kwh = 0.0
    scheduler_overheads: List[float] = []
    action_entropies: List[float] = []

    state = env.reset_state()
    done = False

    while not done:
        cfg = env.cfg

        # 1) Decide which scheduler / action to use
        t0 = time.perf_counter()

        if scheduler_name == "RLMetaDQN":
            # RLMetaDQN chooses among the 5 heuristic schedulers
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32, device=torch_device).unsqueeze(0)
                q_vals = rl_net(s_t)  # type: ignore[arg-type]
                a = int(torch.argmax(q_vals, dim=1).item())
                # Softmax entropy
                probs = torch.softmax(q_vals, dim=1)
                log_probs = torch.log(probs + 1e-9)
                ent = -torch.sum(probs * log_probs, dim=1).item()
                action_entropies.append(float(ent))

            policy_name = SCHEDULER_NAMES[a]
        else:
            # Pure heuristic, no entropy
            policy_name = scheduler_name
            ent = 0.0

        # 2) Simulate dt with current assignments
        finished_before = env._simulate_dt(cfg.dt_s)
        busy_gpus = sum(1 for g in env.gpus if g.job is not None)
        queue_len = len(env.pending)
        power_w = env._power_draw()

        # We don't need the RL reward for metrics, but compute to match semantics
        _ = sim.compute_rl_reward(
            finished_jobs=finished_before,
            busy_gpus=busy_gpus,
            num_gpus=cfg.num_gpus,
            queue_len=queue_len,
            total_power_w=power_w,
            cfg=cfg.reward_cfg,
        )

        # 3) Apply scheduling decision
        env._schedule_jobs(policy_name)

        t1 = time.perf_counter()
        overhead_ms = (t1 - t0) * 1000.0
        scheduler_overheads.append(overhead_ms)

        # 4) Bookkeeping
        queue_lengths.append(float(queue_len))
        gpu_utils.append(float(busy_gpus / cfg.num_gpus if cfg.num_gpus > 0 else 0.0))
        energy_kwh += (power_w * (cfg.dt_s / 3600.0)) / 1000.0

        # 5) Advance time and inject arrivals
        env.time += cfg.dt_s
        env._inject_arrivals()

        # 6) Termination condition
        if env.time >= cfg.horizon_s and not env.pending and \
                all(g.job is None for g in env.gpus):
            env.done = True
        done = env.done

        # 7) Update state
        state = encode_state(
            workload_label=cfg.workload_label,
            intensity_label=cfg.intensity_label,
            num_gpus=cfg.num_gpus,
            num_jobs=len(env.pending),
        )

    # Collect per-job metrics
    for j in env.completed:
        if j.completion_time is None or j.slowdown is None:
            continue
        jcts.append(float(j.completion_time))
        slows.append(float(j.slowdown))
        # SLA success = 1 if NOT violated
        sla_flags.append(1.0 if not j.sla_violated else 0.0)

    if not jcts:
        raise RuntimeError(
            "No completed jobs in simulate_regime(); check env configuration."
        )

    jcts_arr = np.asarray(jcts, dtype=float)
    slows_arr = np.asarray(slows, dtype=float)
    sla_arr = np.asarray(sla_flags, dtype=float)

    qlens_arr = np.asarray(queue_lengths, dtype=float) if queue_lengths else np.array([0.0])
    gpu_arr = np.asarray(gpu_utils, dtype=float) if gpu_utils else np.array([0.0])
    overhead_arr = np.asarray(scheduler_overheads, dtype=float) if scheduler_overheads else np.array([0.0])
    ent_arr = np.asarray(action_entropies, dtype=float) if action_entropies else np.array([0.0])

    avg_jct = float(jcts_arr.mean())
    p95_jct = float(np.percentile(jcts_arr, 95))
    slowdown_mean = float(slows_arr.mean())
    slowdown_p95 = float(np.percentile(slows_arr, 95))
    sla = float(sla_arr.mean())
    gpu_utilization = float(gpu_arr.mean())
    queue_len_mean = float(qlens_arr.mean())
    jct_cv = float(jcts_arr.std(ddof=0) / (jcts_arr.mean() + 1e-9))

    # Jain fairness over JCT
    x = jcts_arr
    fairness_jain = float((x.sum() ** 2) / (len(x) * (np.square(x).sum() + 1e-9)))

    scheduler_overhead_ms = float(overhead_arr.mean())
    action_entropy = float(ent_arr.mean())

    metrics: Dict[str, float] = {
        "avg_jct": avg_jct,
        "p95_jct": p95_jct,
        "slowdown_mean": slowdown_mean,
        "slowdown_p95": slowdown_p95,
        "sla": sla,
        "gpu_utilization": gpu_utilization,
        "energy_kwh": float(energy_kwh),
        "queue_len_mean": queue_len_mean,
        "jct_cv": jct_cv,
        "fairness_jain": fairness_jain,
        "scheduler_overhead_ms": scheduler_overhead_ms,
        "action_entropy": action_entropy,
    }
    return metrics


# ----------------------------------------------------------------------
# Experiment loop
# ----------------------------------------------------------------------

def run_experiments(regimes: Iterable[RegimeConfig], cfg: RunConfig) -> pd.DataFrame:
    regimes = list(regimes)
    rows: List[Dict] = []

    total_regimes = len(regimes)
    total_runs = total_regimes * len(cfg.schedulers) * cfg.num_seeds

    print("=============================================")
    print("Running Core Simulation Experiments")
    print("=============================================")
    print(f"Total regimes: {total_regimes}")
    print(f"Schedulers   : {cfg.schedulers}")
    print(f"Seeds        : {cfg.num_seeds}")
    print(f"Total runs   : {total_runs}")
    print("---------------------------------------------")
    print(f"Output CSV   : {cfg.output_path}")
    print("=============================================\n")

    run_idx = 0

    for regime in regimes:
        for scheduler_name in cfg.schedulers:
            for k in range(cfg.num_seeds):
                run_idx += 1
                seed_global = hash((regime.regime_id(), scheduler_name, k)) % (2**31)
                seed_all(seed_global)

                print(
                    f"[{run_idx:05d}/{total_runs:05d}] "
                    f"regime={regime.regime_id():35s} "
                    f"scheduler={scheduler_name:15s} "
                    f"seed={seed_global}"
                )

                metrics = simulate_regime(
                    workload=regime.workload,
                    intensity=regime.intensity,
                    num_gpus=regime.num_gpus,
                    scheduler_name=scheduler_name,
                    seed=seed_global,
                    super_high_load=regime.super_high_load,
                    rl_ckpt_path=cfg.rl_ckpt_path,
                    device=cfg.device,
                    log_actions=cfg.log_actions,
                )

                # Basic key validation
                expected_keys = [
                    "avg_jct",
                    "p95_jct",
                    "slowdown_mean",
                    "slowdown_p95",
                    "sla",
                    "gpu_utilization",
                    "energy_kwh",
                    "queue_len_mean",
                    "jct_cv",
                    "fairness_jain",
                    "scheduler_overhead_ms",
                    "action_entropy",
                ]
                missing = [k for k in expected_keys if k not in metrics]
                if missing:
                    raise ValueError(
                        f"Missing metrics {missing} for regime={regime.regime_id()}, "
                        f"scheduler={scheduler_name}, seed={seed_global}"
                    )

                row = {
                    "workload": regime.workload,
                    "intensity": regime.intensity,
                    "num_gpus": regime.num_gpus,
                    "super_high_load": int(regime.super_high_load),
                    "regime_id": regime.regime_id(),
                    "scheduler": scheduler_name,
                    "seed": seed_global,
                }
                row.update(metrics)
                rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TPDS core simulation experiments (RLMetaDQN vs heuristics)."
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of seeds per (regime, scheduler). Default: 5.",
    )
    parser.add_argument(
        "--schedulers",
        type=str,
        nargs="+",
        default=[
            "StaticEqual",
            "StaticPriority",
            "EDF",
            "SRPT",
            "DynamicHeuristic",
            "RLMetaDQN",
        ],
        help="Schedulers to evaluate. RLMetaDQN requires --rl-ckpt.",
    )
    parser.add_argument(
        "--rl-ckpt",
        type=str,
        default=None,
        help="Path to rl_meta_dqn_tpds.pt (required for RLMetaDQN).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device for RLMetaDQN ("cpu" or "cuda"). Default: cpu.',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="core_sim_metrics.csv",
        help='Output CSV path. Default: "core_sim_metrics.csv".',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing CSV.",
    )
    parser.add_argument(
        "--no-action-log",
        action="store_true",
        help="Reserved flag (actions are not written to disk here).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output '{output_path}' already exists. Use --overwrite to replace it."
        )

    rl_ckpt_path = (
        Path(args.rl_ckpt).expanduser().resolve() if args.rl_ckpt is not None else None
    )

    cfg = RunConfig(
        num_seeds=args.num_seeds,
        schedulers=args.schedulers,
        rl_ckpt_path=rl_ckpt_path,
        device=args.device,
        output_path=output_path,
        overwrite=args.overwrite,
        log_actions=not args.no_action_log,
    )

    print("=============================================")
    print("Core Simulation Configuration")
    print("=============================================")
    print(cfg.to_json())
    print("=============================================\n")

    regimes = enumerate_core_regimes()
    df = run_experiments(regimes, cfg)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_path, index=False)

    print("\n=============================================")
    print("Done.")
    print(f"Saved metrics to: {cfg.output_path}")
    print("Rows:", len(df))
    print("=============================================")


if __name__ == "__main__":
    main()

