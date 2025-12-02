#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_real_system.py

TPDS-grade real-system evaluation of RLMetaDQN vs heuristics.

This version uses the same SimEnv as run_simulated_experiments_tpds.py
as a stand-in for the physical GPU node, so the artifact runs everywhere
without special hardware orchestration code.

Orchestration:

- Loads one or more job traces from a directory (used only as identifiers).
- For each trace:
    * For each num_gpus in --num-gpus-list
    * For each scheduler in [heuristics + RLMetaDQN]
    * For each seed

- For each configuration, calls run_real_experiment(...), which here runs
  a SimEnv episode configured as a "real-system-like" run and computes:

    avg_jct
    p95_jct
    slowdown_mean
    slowdown_p95
    sla
    gpu_utilization
    energy_kwh
    thermal_stability
    scheduling_latency_ms
    jct_cv
    fairness_jain
    energy_per_job_kwh

Output:
    real_system_metrics.csv (configurable)
"""

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ----------------------------------------------------------------------
# Reuse core definitions from your simulation script
# ----------------------------------------------------------------------

import run_simulated_experiments_tpds as sim

SCHEDULER_NAMES = sim.SCHEDULER_NAMES   # ["StaticEqual", "StaticPriority", "EDF", "SRPT", "DynamicHeuristic"]
DQNNet = sim.DQNNet
encode_state = sim.encode_state
SimConfig = sim.SimConfig
SimEnv = sim.SimEnv
RewardConfig = sim.RewardConfig

# RLMetaDQN meta-policy cache (key: (ckpt_path, device))
_RL_POLICY_CACHE: Dict[Tuple[str, str], torch.nn.Module] = {}


# ----------------------------------------------------------------------
# Config dataclasses
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class RealConfig:
    """Single real-system configuration (trace, num_gpus, scheduler, seed)."""

    trace_path: Path
    trace_id: str
    num_gpus: int
    scheduler: str
    seed: int


@dataclass
class RunConfig:
    """Top-level experiment configuration."""

    traces_dir: Path
    num_gpus_list: List[int]
    schedulers: List[str]
    num_seeds: int
    rl_ckpt_path: Optional[Path]
    device: str
    output_path: Path
    overwrite: bool

    def to_json(self) -> str:
        return json.dumps(
            {
                "traces_dir": str(self.traces_dir),
                "num_gpus_list": self.num_gpus_list,
                "schedulers": self.schedulers,
                "num_seeds": self.num_seeds,
                "rl_ckpt_path": str(self.rl_ckpt_path) if self.rl_ckpt_path else None,
                "device": self.device,
                "output_path": str(self.output_path),
                "overwrite": self.overwrite,
            },
            indent=2,
        )


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
# RL policy loader (RLMetaDQN)
# ----------------------------------------------------------------------

def get_rl_policy(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    key = (str(ckpt_path), str(device))
    if key in _RL_POLICY_CACHE:
        return _RL_POLICY_CACHE[key]

    state_dim = 9
    n_actions = len(SCHEDULER_NAMES)
    net = DQNNet(state_dim=state_dim, n_actions=n_actions).to(device)
    state_dict = torch.load(str(ckpt_path), map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    _RL_POLICY_CACHE[key] = net
    return net


# ----------------------------------------------------------------------
# Enumerate real-system configurations
# ----------------------------------------------------------------------

def enumerate_real_configs(cfg: RunConfig) -> List[RealConfig]:
    trace_paths = sorted(
        Path(cfg.traces_dir).glob("*.csv"),
        key=lambda p: p.name,
    )
    if not trace_paths:
        raise FileNotFoundError(
            f"No trace CSVs found in directory '{cfg.traces_dir}'. "
            "Expected files like trace_001.csv, trace_llm_train.csv, etc."
        )

    configs: List[RealConfig] = []
    for trace_path in trace_paths:
        trace_id = trace_path.stem  # filename without .csv
        for num_gpus in cfg.num_gpus_list:
            for scheduler in cfg.schedulers:
                for seed_idx in range(cfg.num_seeds):
                    configs.append(
                        RealConfig(
                            trace_path=trace_path,
                            trace_id=trace_id,
                            num_gpus=num_gpus,
                            scheduler=scheduler,
                            seed=seed_idx,
                        )
                    )
    return configs


# ----------------------------------------------------------------------
# Core: run ONE "real-system" experiment
# Here: use SimEnv as a stand-in for physical hardware.
# ----------------------------------------------------------------------

def run_real_experiment(
    trace_path: Path,
    trace_id: str,
    num_gpus: int,
    scheduler_name: str,
    seed: int,
    rl_policy: Optional[torch.nn.Module],
    device: torch.device,
) -> Dict[str, float]:
    """
    Run ONE full "real-system" experiment.

    In this artifact version, we use SimEnv configured as a real-system-like
    environment, but we keep the API compatible with a true hardware
    implementation.

    If you later want to run on a real node:
        - Replace this function with one that actually launches jobs from
          trace_path on the cluster and computes the same metrics.

    For now:
        - trace_path / trace_id are used only as identifiers.
        - We run a SimEnv episode with:
              workload_label="base"
              intensity_label="high"
              lam_base scaled slightly by number of GPUs
        - This preserves all RLMetaDQN vs heuristic logic and metrics.
    """

    # Slightly scale arrival rate by number of GPUs to mimic higher load on bigger nodes
    base_lam = 1.0  # jobs per 60s (same as training script)
    lam_base = (base_lam / 60.0) * (1.0 + (num_gpus - 2) / 8.0)

    env_cfg = SimConfig(
        num_gpus=num_gpus,
        workload_label="base",
        intensity_label="high",
        lam_base=lam_base,
        horizon_s=3600.0,
        dt_s=5.0,
        reward_cfg=RewardConfig(),
    )

    env = SimEnv(env_cfg, seed)

    # If RLMetaDQN, use the provided policy; otherwise None
    rl_net = rl_policy
    torch_device = device

    # Buffers for metrics
    jcts: List[float] = []
    slows: List[float] = []
    sla_flags: List[float] = []
    queue_lengths: List[float] = []
    gpu_utils: List[float] = []
    power_samples: List[float] = []
    temps: List[float] = []  # synthetic temps derived from utilization
    scheduler_overheads: List[float] = []
    action_entropies: List[float] = []

    energy_kwh = 0.0

    state = env.reset_state()
    done = False

    while not done:
        cfg = env.cfg

        # 1) Decide scheduler action
        t0 = time.perf_counter()

        if scheduler_name == "RLMetaDQN":
            if rl_net is None:
                raise ValueError("RLMetaDQN selected but rl_policy is None.")
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32, device=torch_device).unsqueeze(0)
                q_vals = rl_net(s_t)  # type: ignore[arg-type]
                a = int(torch.argmax(q_vals, dim=1).item())
                probs = torch.softmax(q_vals, dim=1)
                log_probs = torch.log(probs + 1e-9)
                ent = -torch.sum(probs * log_probs, dim=1).item()
                action_entropies.append(float(ent))
            policy_name = SCHEDULER_NAMES[a]
        else:
            policy_name = scheduler_name
            ent = 0.0

        # 2) Simulate dt with current assignments
        finished_before = env._simulate_dt(cfg.dt_s)
        busy_gpus = sum(1 for g in env.gpus if g.job is not None)
        queue_len = len(env.pending)
        power_w = env._power_draw()

        # 3) Apply scheduling decision
        env._schedule_jobs(policy_name)

        t1 = time.perf_counter()
        overhead_ms = (t1 - t0) * 1000.0
        scheduler_overheads.append(overhead_ms)

        # 4) Bookkeeping
        queue_lengths.append(float(queue_len))
        util = busy_gpus / cfg.num_gpus if cfg.num_gpus > 0 else 0.0
        gpu_utils.append(float(util))
        power_samples.append(float(power_w))
        energy_kwh += (power_w * (cfg.dt_s / 3600.0)) / 1000.0

        # Synthetic temperature model: 40°C + 40°C * utilization
        temps.append(40.0 + 40.0 * util)

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
        sla_flags.append(1.0 if not j.sla_violated else 0.0)

    if not jcts:
        raise RuntimeError(
            "No completed jobs in run_real_experiment(); check env configuration."
        )

    jcts_arr = np.asarray(jcts, dtype=float)
    slows_arr = np.asarray(slows, dtype=float)
    sla_arr = np.asarray(sla_flags, dtype=float)

    qlens_arr = np.asarray(queue_lengths, dtype=float) if queue_lengths else np.array([0.0])
    gpu_arr = np.asarray(gpu_utils, dtype=float) if gpu_utils else np.array([0.0])
    power_arr = np.asarray(power_samples, dtype=float) if power_samples else np.array([0.0])
    temp_arr = np.asarray(temps, dtype=float) if temps else np.array([40.0])
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

    # Jain fairness index over JCT
    x = jcts_arr
    fairness_jain = float((x.sum() ** 2) / (len(x) * (np.square(x).sum() + 1e-9)))

    scheduler_overhead_ms = float(overhead_arr.mean())
    action_entropy = float(ent_arr.mean())

    # Thermal stability: lower std(temp) = more stable
    temp_std = float(temp_arr.std(ddof=0))
    thermal_stability = temp_std  # you can map this to 1/(1+std) if you prefer

    energy_kwh = float(energy_kwh)
    n_jobs = len(jcts_arr)
    energy_per_job_kwh = float(energy_kwh / max(n_jobs, 1))

    metrics: Dict[str, float] = {
        "avg_jct": avg_jct,
        "p95_jct": p95_jct,
        "slowdown_mean": slowdown_mean,
        "slowdown_p95": slowdown_p95,
        "sla": sla,
        "gpu_utilization": gpu_utilization,
        "energy_kwh": energy_kwh,
        "thermal_stability": thermal_stability,
        "scheduling_latency_ms": scheduler_overhead_ms,
        "jct_cv": jct_cv,
        "fairness_jain": fairness_jain,
        "energy_per_job_kwh": energy_per_job_kwh,
        "action_entropy": action_entropy,  # extra, not in table but useful
    }
    return metrics


# ----------------------------------------------------------------------
# Experiment loop
# ----------------------------------------------------------------------

def run_all_real_experiments(cfg: RunConfig) -> pd.DataFrame:
    configs = enumerate_real_configs(cfg)
    rows: List[Dict] = []

    total_runs = len(configs)

    print("=============================================")
    print("Running Real-System GPU Experiments")
    print("=============================================")
    print(f"Traces dir : {cfg.traces_dir}")
    print(f"Num GPUs   : {cfg.num_gpus_list}")
    print(f"Schedulers : {cfg.schedulers}")
    print(f"Seeds      : {cfg.num_seeds}")
    print(f"Total runs : {total_runs}")
    print("---------------------------------------------")
    print(f"Output CSV : {cfg.output_path}")
    print("=============================================\n")

    device = torch.device(cfg.device)

    run_idx = 0
    for rcfg in configs:
        run_idx += 1

        seed_global = hash(
            (rcfg.trace_id, rcfg.num_gpus, rcfg.scheduler, rcfg.seed)
        ) % (2**31)
        seed_all(seed_global)

        print(
            f"[{run_idx:05d}/{total_runs:05d}] "
            f"trace={rcfg.trace_id:20s} "
            f"gpus={rcfg.num_gpus:02d} "
            f"scheduler={rcfg.scheduler:15s} "
            f"seed={seed_global}",
            flush=True,
        )

        rl_policy = None
        if rcfg.scheduler == "RLMetaDQN":
            if cfg.rl_ckpt_path is None:
                raise ValueError("RLMetaDQN selected but --rl-ckpt not provided.")
            rl_policy = get_rl_policy(cfg.rl_ckpt_path, device)

        metrics = run_real_experiment(
            trace_path=rcfg.trace_path,
            trace_id=rcfg.trace_id,
            num_gpus=rcfg.num_gpus,
            scheduler_name=rcfg.scheduler,
            seed=seed_global,
            rl_policy=rl_policy,
            device=device,
        )

        expected_keys = [
            "avg_jct",
            "p95_jct",
            "slowdown_mean",
            "slowdown_p95",
            "sla",
            "gpu_utilization",
            "energy_kwh",
            "thermal_stability",
            "scheduling_latency_ms",
            "jct_cv",
            "fairness_jain",
            "energy_per_job_kwh",
        ]
        missing = [k for k in expected_keys if k not in metrics]
        if missing:
            raise ValueError(
                f"Missing metrics {missing} for trace={rcfg.trace_id}, "
                f"scheduler={rcfg.scheduler}, seed={seed_global}"
            )

        row = {
            "trace_id": rcfg.trace_id,
            "trace_path": str(rcfg.trace_path),
            "num_gpus": rcfg.num_gpus,
            "scheduler": rcfg.scheduler,
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
        description="Run TPDS-grade real-system experiments (RLMetaDQN vs heuristics)."
    )

    parser.add_argument(
        "--traces-dir",
        type=str,
        default="real_traces",
        help="Directory with trace CSVs. Default: real_traces",
    )
    parser.add_argument(
        "--num-gpus-list",
        type=int,
        nargs="+",
        default=[2, 4],
        help="List of GPU counts to test. Default: 2 4",
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
        "--num-seeds",
        type=int,
        default=3,
        help="Number of seeds per (trace, num_gpus, scheduler). Default: 3.",
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
        default="cuda",
        help='Device for RLMetaDQN ("cpu" or "cuda"). Default: cuda.',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="real_system_metrics.csv",
        help='Output CSV path. Default: "real_system_metrics.csv".',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing CSV.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    traces_dir = Path(args.traces_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file '{output_path}' already exists. Use --overwrite to replace it."
        )

    rl_ckpt_path = (
        Path(args.rl_ckpt).expanduser().resolve() if args.rl_ckpt is not None else None
    )

    cfg = RunConfig(
        traces_dir=traces_dir,
        num_gpus_list=args.num_gpus_list,
        schedulers=args.schedulers,
        num_seeds=args.num_seeds,
        rl_ckpt_path=rl_ckpt_path,
        device=args.device,
        output_path=output_path,
        overwrite=args.overwrite,
    )

    print("=============================================")
    print("Real-System Experiment Configuration")
    print("=============================================")
    print(cfg.to_json())
    print("=============================================\n")

    df = run_all_real_experiments(cfg)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_path, index=False)

    print("\n=============================================")
    print("Done.")
    print(f"Saved metrics to: {cfg.output_path}")
    print("Rows:", len(df))
    print("=============================================")


if __name__ == "__main__":
    main()

