#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_robustness.py

Robustness Experiments (Drift + Bandits) for TPDS.

This script evaluates how well different schedulers and online bandit
meta-policies adapt to non-stationary workload regimes.

Scenarios
---------
1) drift_base_bursty_diurnal
   - Phases: base → bursty → diurnal (smooth regime shifts)

2) drift_diurnal_shifted_bursty
   - Phases: diurnal → shifted_diurnal → bursty

3) abrupt_spike_base_burst_diurnal
   - Phases: base → bursty (high intensity spike) → diurnal

Methods
-------
Heuristics (fixed scheduler):
    StaticEqual, StaticPriority, EDF, SRPT, DynamicHeuristic

Bandits (online heuristic selection over SCHEDULER_NAMES):
    BanditEpsGreedy
    BanditEXP3
    BanditLinUCB

Metrics (per scenario, method, seed)
------------------------------------
    avg_jct
    p95_jct
    slowdown_mean
    slowdown_p95
    sla                       # fraction of jobs meeting SLA
    gpu_utilization           # average utilization (0–1)
    energy_kwh
    queue_len_mean
    jct_cv                    # coefficient of variation of JCT
    fairness_jain             # Jain index over JCT (on normalized values)
    scheduler_overhead_ms     # average decision overhead
    action_entropy            # mean entropy over action distributions
    transition_recovery_time  # mean time to queue recovery after drifts

Output
------
    robustness_metrics.csv  (configurable via --output)
"""

import argparse
import hashlib
import math
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Force line-buffered stdout so progress prints appear immediately
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    # Older Pythons may not have reconfigure; safe to ignore
    pass

# ----------------------------------------------------------------------
# Import core simulation building blocks from your existing script
# ----------------------------------------------------------------------

import run_simulated_experiments_tpds as sim

SimJob = sim.SimJob
SimGpu = sim.SimGpu
RewardConfig = sim.RewardConfig
compute_rl_reward = sim.compute_rl_reward
pick_job_heuristic = sim.pick_job_heuristic
SCHEDULER_NAMES = sim.SCHEDULER_NAMES  # ["StaticEqual", "StaticPriority", "EDF", "SRPT", "DynamicHeuristic"]


# ----------------------------------------------------------------------
# Drift environment
# ----------------------------------------------------------------------

@dataclass
class DriftPhase:
    workload_label: str        # "base", "bursty", "diurnal", "shifted_diurnal"
    intensity_label: str       # "low", "medium", "high"
    duration_s: float          # duration of this phase in seconds


@dataclass
class DriftEnvConfig:
    num_gpus: int
    phases: List[DriftPhase]
    lam_base: float            # base arrival rate per second (scaled by intensity)
    dt_s: float = 5.0
    power_idle_w: float = 100.0
    power_per_gpu_w: float = 250.0
    reward_cfg: RewardConfig = RewardConfig()


class DriftEnv:
    """
    Non-stationary environment with piecewise phases.
    """

    def __init__(self, cfg: DriftEnvConfig, seed: int):
        self.cfg = cfg
        self.seed = seed

        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.time = 0.0
        self.total_horizon = sum(p.duration_s for p in cfg.phases)

        self.job_counter = 0
        self.pending: List[SimJob] = []
        self.completed: List[SimJob] = []

        self.gpus: List[SimGpu] = [SimGpu(gpu_id=i) for i in range(cfg.num_gpus)]
        self.done = False

    # ---- helpers -----------------------------------------------------

    def _current_phase_index(self) -> int:
        t = self.time
        acc = 0.0
        for idx, ph in enumerate(self.cfg.phases):
            acc += ph.duration_s
            if t < acc:
                return idx
        return len(self.cfg.phases) - 1

    def _current_phase(self) -> DriftPhase:
        return self.cfg.phases[self._current_phase_index()]

    def _current_lambda(self) -> float:
        """
        Lambda(t) in jobs/second, depending on workload & intensity.
        """
        phase = self._current_phase()
        lam0 = self.cfg.lam_base

        # intensity scaling (same semantics as original script)
        if phase.intensity_label == "low":
            lam_int = lam0
        elif phase.intensity_label == "medium":
            lam_int = 2.0 * lam0
        else:  # "high"
            lam_int = 4.0 * lam0

        # workload shape
        t = self.time
        horizon = self.total_horizon
        wl = phase.workload_label

        if wl == "base":
            lam = lam_int
        elif wl == "bursty":
            period = 300.0
            phase_id = int(t // period)
            lam = lam_int * (3.0 if phase_id % 2 == 0 else 0.3)
        elif wl == "diurnal":
            x = 2 * math.pi * (t / horizon)
            lam = lam_int * (0.5 + 0.5 * (1 + math.sin(x)))
        elif wl == "shifted_diurnal":
            x = 2 * math.pi * ((t + horizon / 4) / horizon)
            lam = lam_int * (0.5 + 0.5 * (1 + math.sin(x)))
        else:
            lam = lam_int

        return max(lam, 1e-6)

    def _inject_arrivals(self):
        """
        Inject new jobs for the current dt.

        IMPORTANT FIX:
        --------------
        Do NOT inject new jobs once self.time >= self.total_horizon.
        Otherwise, the queue never empties and the env never terminates.
        """
        if self.time >= self.total_horizon:
            return

        lam_t = self._current_lambda()      # jobs/s
        mu = lam_t * self.cfg.dt_s          # expected arrivals in dt
        n_new = np.random.poisson(mu)

        for _ in range(n_new):
            est_runtime_s = self.rng.uniform(30.0, 900.0)
            deadline_factor = self.rng.uniform(1.2, 2.0)
            deadline_s = est_runtime_s * deadline_factor
            priority = self.rng.randint(0, 3)

            job = SimJob(
                job_id=self.job_counter,
                submit_time=self.time,
                est_runtime_s=est_runtime_s,
                deadline_s=deadline_s,
                priority=priority,
                remaining_s=est_runtime_s,
            )
            self.job_counter += 1
            self.pending.append(job)

    def _simulate_dt(self, dt: float) -> List[sim.FinishedJobInfo]:
        finished_this_step: List[sim.FinishedJobInfo] = []

        for g in self.gpus:
            if g.job is not None:
                g.job.remaining_s -= dt
                if g.job.remaining_s <= 0.0:
                    g.job.end_time = self.time
                    deadline_abs = g.job.submit_time + g.job.deadline_s
                    g.job.sla_violated = g.job.end_time > deadline_abs
                    self.completed.append(g.job)
                    finished_this_step.append(
                        sim.FinishedJobInfo(
                            submit_time=g.job.submit_time,
                            start_time=g.job.start_time or g.job.submit_time,
                            finish_time=g.job.end_time,
                            est_runtime=g.job.est_runtime_s,
                            sla_missed=g.job.sla_violated,
                        )
                    )
                    g.job = None
        return finished_this_step

    def _schedule_jobs(self, policy: str):
        free_gpus = [g for g in self.gpus if g.job is None]
        while free_gpus and self.pending:
            g = free_gpus.pop(0)
            idx = pick_job_heuristic(self.pending, policy, self.time)
            job = self.pending.pop(idx)
            job.start_time = self.time
            job.remaining_s = job.est_runtime_s
            g.job = job

    def _power_draw(self) -> float:
        busy_gpus = sum(1 for g in self.gpus if g.job is not None)
        return self.cfg.power_idle_w + busy_gpus * self.cfg.power_per_gpu_w

    # ---- public API --------------------------------------------------

    def reset(self):
        self.time = 0.0
        self.job_counter = 0
        self.pending = []
        self.completed = []
        for g in self.gpus:
            g.job = None
        self.done = False

    def step(self, policy: str) -> Tuple[float, int, float, bool]:
        """
        Advance the environment by dt using the given heuristic policy.
        """
        cfg = self.cfg

        # arrivals + progress on already running jobs
        self._inject_arrivals()
        finished_before = self._simulate_dt(cfg.dt_s)

        busy_gpus = sum(1 for g in self.gpus if g.job is not None)
        queue_len = len(self.pending)
        power_w = self._power_draw()

        reward_rl = compute_rl_reward(
            finished_jobs=finished_before,
            busy_gpus=busy_gpus,
            num_gpus=cfg.num_gpus,
            queue_len=queue_len,
            total_power_w=power_w,
            cfg=cfg.reward_cfg,
        )

        # apply scheduling policy
        self._schedule_jobs(policy)

        # advance time
        self.time += cfg.dt_s

        # termination condition:
        #  - horizon reached AND
        #  - no pending jobs AND
        #  - all GPUs idle
        if self.time >= self.total_horizon and not self.pending and \
                all(g.job is None for g in self.gpus):
            self.done = True

        return reward_rl, queue_len, float(busy_gpus), self.done

    def current_phase_index(self) -> int:
        return self._current_phase_index()


# ----------------------------------------------------------------------
# Bandit algorithms
# ----------------------------------------------------------------------

class EpsGreedyBandit:
    def __init__(self, n_actions: int, eps_start: float = 0.3,
                 eps_end: float = 0.05, decay_steps: int = 5000):
        self.n_actions = n_actions
        self.q = np.zeros(n_actions, dtype=np.float64)
        self.counts = np.zeros(n_actions, dtype=np.int64)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_steps = max(decay_steps, 1)
        self.t = 0

    def epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -self.t / self.decay_steps
        )

    def select_action(self) -> Tuple[int, np.ndarray]:
        self.t += 1
        eps = self.epsilon()
        if random.random() < eps:
            a = random.randrange(self.n_actions)
        else:
            a = int(np.argmax(self.q))

        # action distribution (for entropy)
        p = np.full(self.n_actions, eps / self.n_actions, dtype=np.float64)
        best = int(np.argmax(self.q))
        p[best] += (1.0 - eps)
        return a, p

    def update(self, a: int, r: float):
        self.counts[a] += 1
        alpha = 1.0 / self.counts[a]
        self.q[a] += alpha * (r - self.q[a])


class EXP3Bandit:
    """
    EXP3 with extra guards to ensure p sums to 1 and weights never overflow.
    """

    def __init__(self, n_actions: int, gamma: float = 0.1):
        self.n_actions = n_actions
        self.gamma = gamma
        self.w = np.ones(n_actions, dtype=np.float64)
        self.t = 0

    def probs(self) -> np.ndarray:
        """
        Numerically stable probability computation.
        """
        w = self.w
        w = np.where(np.isfinite(w), w, 1.0)
        w_sum = w.sum()
        if (not np.isfinite(w_sum)) or (w_sum <= 0.0):
            w = np.ones(self.n_actions, dtype=np.float64)
            w_sum = w.sum()
        self.w = w

        base = w / w_sum
        p = (1.0 - self.gamma) * base + self.gamma / self.n_actions

        # Clip and renormalize to avoid "probabilities do not sum to 1"
        p = np.clip(p, 1e-12, 1.0)
        p_sum = p.sum()
        if (not np.isfinite(p_sum)) or (p_sum <= 0.0):
            p = np.ones(self.n_actions, dtype=np.float64) / self.n_actions
        else:
            p /= p_sum

        return p

    def select_action(self) -> Tuple[int, np.ndarray]:
        self.t += 1
        p = self.probs()
        a = int(np.random.choice(self.n_actions, p=p))
        return a, p

    def update(self, a: int, r: float):
        """
        r is assumed normalized in [0, 1]; importance-weighted update.

        Strong clipping to prevent weight explosion.
        """
        p_all = self.probs()
        p = float(p_all[a])
        p = max(p, 1e-6)  # avoid division by near-zero

        est_reward = r / p
        est_reward = float(np.clip(est_reward, -20.0, 20.0))

        eta = self.gamma / self.n_actions
        try:
            factor = math.exp(eta * est_reward)
        except OverflowError:
            factor = float("inf")

        if (not np.isfinite(factor)) or (factor <= 0.0):
            self.w = np.ones(self.n_actions, dtype=np.float64)
        else:
            new_w = self.w[a] * factor
            if (not np.isfinite(new_w)) or (new_w > 1e100):
                self.w = np.ones(self.n_actions, dtype=np.float64)
            else:
                self.w[a] = new_w
                self.w = np.clip(self.w, 1e-12, 1e100)


class LinUCBBandit:
    """
    Simple contextual LinUCB over a fixed feature vector x_t.
    """

    def __init__(self, n_actions: int, d: int, alpha: float = 1.0):
        self.n_actions = n_actions
        self.d = d
        self.alpha = alpha

        self.A = [np.eye(d, dtype=np.float64) for _ in range(n_actions)]
        self.b = [np.zeros((d, 1), dtype=np.float64) for _ in range(n_actions)]

    def select_action(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        x = x.reshape(-1, 1)
        p_values = np.zeros(self.n_actions, dtype=np.float64)

        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mu = (theta.T @ x).item()
            sigma = np.sqrt((x.T @ A_inv @ x).item())
            p_values[a] = mu + self.alpha * sigma

        a = int(np.argmax(p_values))
        probs = np.zeros(self.n_actions, dtype=np.float64)
        probs[a] = 1.0
        return a, probs

    def update(self, a: int, x: np.ndarray, r: float):
        x = x.reshape(-1, 1)
        self.A[a] += x @ x.T
        self.b[a] += r * x


# ----------------------------------------------------------------------
# Experiment utilities
# ----------------------------------------------------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_deterministic_seed(scenario_name: str, method: str, seed_idx: int) -> int:
    s = f"{scenario_name}|{method}|{seed_idx}"
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def normalize_reward_for_bandit(r: float, cfg: RewardConfig) -> float:
    r_clipped = max(cfg.clip_min, min(cfg.clip_max, r))
    return (r_clipped - cfg.clip_min) / (cfg.clip_max - cfg.clip_min + 1e-9)


def compute_metrics_from_env(
    env: DriftEnv,
    queue_lengths: List[int],
    busy_gpus_ts: List[float],
    action_entropies: List[float],
    scheduler_overheads: List[float],
    cfg: DriftEnvConfig,
) -> Dict[str, float]:
    """
    Aggregate metrics. All heavy NumPy ops are wrapped in np.errstate to
    suppress benign overflow warnings; numbers are clipped where needed.
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
            raise RuntimeError("No completed jobs in DriftEnv run.")

        jcts_arr = np.asarray(jcts, dtype=float)
        slows_arr = np.asarray(slows, dtype=float)
        sla_arr = np.asarray(sla_flags, dtype=float)

        queue_arr = np.asarray(queue_lengths, dtype=float) if queue_lengths else np.array([0.0])
        busy_arr = np.asarray(busy_gpus_ts, dtype=float) if busy_gpus_ts else np.array([0.0])
        overhead_arr = np.asarray(scheduler_overheads, dtype=float) if scheduler_overheads else np.array([0.0])
        ent_arr = np.asarray(action_entropies, dtype=float) if action_entropies else np.array([0.0])

        avg_jct = float(jcts_arr.mean())
        p95_jct = float(np.percentile(jcts_arr, 95))
        slowdown_mean = float(slows_arr.mean())
        slowdown_p95 = float(np.percentile(slows_arr, 95))
        sla = float(sla_arr.mean())

        gpu_utilization = float((busy_arr / max(cfg.num_gpus, 1)).mean())
        queue_len_mean = float(queue_arr.mean())
        jct_cv = float(jcts_arr.std(ddof=0) / (avg_jct + 1e-9))

        x = jcts_arr
        x_mean = x.mean() + 1e-9
        x_norm = x / x_mean
        x_norm = np.clip(x_norm, -1e6, 1e6)

        num = (x_norm.sum() ** 2)
        den = len(x_norm) * (np.square(x_norm).sum() + 1e-9)
        fairness_jain_raw = num / den if den > 0 else 0.0
        fairness_jain = float(min(max(fairness_jain_raw, 0.0), 1.0))

        scheduler_overhead_ms = float(overhead_arr.mean())
        action_entropy = float(ent_arr.mean())

        energy_kwh = 0.0
        for busy in busy_gpus_ts:
            power_w = cfg.power_idle_w + busy * cfg.power_per_gpu_w
            energy_kwh += (power_w * (cfg.dt_s / 3600.0)) / 1000.0
        energy_kwh = float(energy_kwh)

        dt = cfg.dt_s
        times = np.arange(len(queue_lengths)) * dt
        phase_durations = [ph.duration_s for ph in cfg.phases]
        phase_bounds = np.cumsum(phase_durations)[:-1]

        rec_times: List[float] = []
        for T in phase_bounds:
            idx0 = int(np.searchsorted(times, T))
            if idx0 >= len(queue_lengths):
                continue
            q_after = queue_lengths[idx0:]
            if not q_after:
                continue
            q_max = max(q_after)
            threshold = 0.5 * q_max
            rec_idx = None
            for j, q in enumerate(q_after):
                if q <= threshold:
                    rec_idx = idx0 + j
                    break
            if rec_idx is not None:
                rec_time = times[rec_idx] - T
            else:
                rec_time = times[-1] - T
            rec_times.append(float(rec_time))

        transition_recovery_time = float(np.mean(rec_times)) if rec_times else 0.0

    return {
        "avg_jct": avg_jct,
        "p95_jct": p95_jct,
        "slowdown_mean": slowdown_mean,
        "slowdown_p95": slowdown_p95,
        "sla": sla,
        "gpu_utilization": gpu_utilization,
        "energy_kwh": energy_kwh,
        "queue_len_mean": queue_len_mean,
        "jct_cv": jct_cv,
        "fairness_jain": fairness_jain,
        "scheduler_overhead_ms": scheduler_overhead_ms,
        "action_entropy": action_entropy,
        "transition_recovery_time": transition_recovery_time,
    }


# ----------------------------------------------------------------------
# Robustness experiment runner (single configuration)
# ----------------------------------------------------------------------

def run_single_method(
    scenario_name: str,
    phases: List[DriftPhase],
    method: str,
    num_gpus: int,
    seed_idx: int,
) -> Dict[str, float]:
    seed_global = make_deterministic_seed(scenario_name, method, seed_idx)
    seed_all(seed_global)

    # START log for this worker task
    print(f"[START] scenario={scenario_name} method={method} seed_idx={seed_idx}", flush=True)

    base_lam = 1.0  # jobs per 60s
    lam_base = base_lam / 60.0

    cfg_env = DriftEnvConfig(
        num_gpus=num_gpus,
        phases=phases,
        lam_base=lam_base,
        dt_s=5.0,
        power_idle_w=100.0,
        power_per_gpu_w=250.0,
        reward_cfg=RewardConfig(),
    )

    env = DriftEnv(cfg_env, seed_global)
    env.reset()

    queue_lengths: List[int] = []
    busy_gpus_ts: List[float] = []
    scheduler_overheads: List[float] = []
    action_entropies: List[float] = []

    n_actions = len(SCHEDULER_NAMES)
    bandit: Optional[object] = None
    method_type = "heuristic"

    if method == "BanditEpsGreedy":
        bandit = EpsGreedyBandit(n_actions=n_actions)
        method_type = "bandit"
    elif method == "BanditEXP3":
        bandit = EXP3Bandit(n_actions=n_actions)
        method_type = "bandit"
    elif method == "BanditLinUCB":
        bandit = LinUCBBandit(n_actions=n_actions, d=5, alpha=1.0)
        method_type = "bandit"

    done = False
    max_steps = int((env.total_horizon / cfg_env.dt_s) * 10)  # generous safety cap
    steps = 0

    while not done and steps < max_steps:
        t0 = time.perf_counter()

        if method_type == "heuristic":
            policy_name = method
            probs = np.ones(n_actions, dtype=np.float64) / n_actions
            a = -1
        else:
            if isinstance(bandit, LinUCBBandit):
                phase_idx = env.current_phase_index()
                phase_one_hot = np.zeros(3, dtype=np.float32)
                phase_one_hot[min(phase_idx, 2)] = 1.0
                queue_norm = math.tanh(len(env.pending) / 50.0) if env.pending else 0.0
                busy = sum(1 for g in env.gpus if g.job is not None)
                util = busy / max(cfg_env.num_gpus, 1)
                x = np.concatenate(
                    [phase_one_hot, np.array([queue_norm, util], dtype=np.float32)],
                    axis=0,
                )
                a, probs = bandit.select_action(x)
            else:
                a, probs = bandit.select_action()  # type: ignore[arg-type]

            policy_name = SCHEDULER_NAMES[a]

        reward_rl, qlen, busy, done = env.step(policy_name)

        if method_type == "bandit":
            r_bandit = normalize_reward_for_bandit(reward_rl, cfg_env.reward_cfg)
            if isinstance(bandit, LinUCBBandit):
                phase_idx = env.current_phase_index()
                phase_one_hot = np.zeros(3, dtype=np.float32)
                phase_one_hot[min(phase_idx, 2)] = 1.0
                queue_norm = math.tanh(len(env.pending) / 50.0) if env.pending else 0.0
                busy_update = sum(1 for g in env.gpus if g.job is not None)
                util_update = busy_update / max(cfg_env.num_gpus, 1)
                x = np.concatenate(
                    [phase_one_hot, np.array([queue_norm, util_update], dtype=np.float32)],
                    axis=0,
                )
                bandit.update(a, x, r_bandit)  # type: ignore[arg-type]
            else:
                bandit.update(a, r_bandit)  # type: ignore[arg-type]

        t1 = time.perf_counter()
        scheduler_overheads.append((t1 - t0) * 1000.0)

        queue_lengths.append(int(qlen))
        busy_gpus_ts.append(float(busy))

        probs = np.asarray(probs, dtype=np.float64)
        probs = np.clip(probs, 1e-9, 1.0)
        ent = -float(np.sum(probs * np.log(probs)))
        action_entropies.append(ent)

        steps += 1

    if not env.done:
        print(
            f"[WARN] scenario={scenario_name} method={method} seed_idx={seed_idx} "
            f"hit max_steps={max_steps} without env.done=True (forcing stop).",
            flush=True,
        )

    metrics = compute_metrics_from_env(
        env=env,
        queue_lengths=queue_lengths,
        busy_gpus_ts=busy_gpus_ts,
        action_entropies=action_entropies,
        scheduler_overheads=scheduler_overheads,
        cfg=cfg_env,
    )

    out_row = {
        "scenario": scenario_name,
        "method": method,
        "num_gpus": num_gpus,
        "seed": seed_global,
    }
    out_row.update(metrics)

    # DONE log for this worker task
    print(f"[DONE ] scenario={scenario_name} method={method} seed_idx={seed_idx}", flush=True)

    return out_row


# ----------------------------------------------------------------------
# Scenarios definition
# ----------------------------------------------------------------------

def get_scenarios() -> Dict[str, List[DriftPhase]]:
    T = 1200.0

    scenarios = {
        "drift_base_bursty_diurnal": [
            DriftPhase("base", "medium", T),
            DriftPhase("bursty", "medium", T),
            DriftPhase("diurnal", "medium", T),
        ],
        "drift_diurnal_shifted_bursty": [
            DriftPhase("diurnal", "medium", T),
            DriftPhase("shifted_diurnal", "medium", T),
            DriftPhase("bursty", "medium", T),
        ],
        "abrupt_spike_base_burst_diurnal": [
            DriftPhase("base", "medium", T),
            DriftPhase("bursty", "high", T / 2),
            DriftPhase("diurnal", "medium", 1.5 * T),
        ],
    }
    return scenarios


# ----------------------------------------------------------------------
# CLI and main
# ----------------------------------------------------------------------

@dataclass
class RunConfig:
    num_seeds: int
    num_gpus: int
    methods: List[str]
    output_path: Path
    overwrite: bool

    def to_json(self) -> str:
        return (
            "{\n"
            f'  "num_seeds": {self.num_seeds},\n'
            f'  "num_gpus": {self.num_gpus},\n'
            f'  "methods": {self.methods},\n'
            f'  "output_path": "{self.output_path}",\n'
            f'  "overwrite": {self.overwrite!r}\n'
            "}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run robustness experiments (drift + bandits) for RLMetaDQN vs heuristics."
    )

    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of seeds per (scenario, method).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs for robustness scenarios (simulated).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=[
            "StaticEqual",
            "StaticPriority",
            "EDF",
            "SRPT",
            "DynamicHeuristic",
            "BanditEpsGreedy",
            "BanditEXP3",
            "BanditLinUCB",
        ],
        help="Methods to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="robustness_metrics.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing CSV.",
    )

    return parser.parse_args()


def _worker(task: Tuple[str, List[DriftPhase], str, int, int]) -> Dict[str, float]:
    scenario_name, phases, method, num_gpus, seed_idx = task
    return run_single_method(
        scenario_name=scenario_name,
        phases=phases,
        method=method,
        num_gpus=num_gpus,
        seed_idx=seed_idx,
    )


def main() -> None:
    args = parse_args()

    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file '{output_path}' already exists. Use --overwrite to replace it."
        )

    cfg = RunConfig(
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        methods=args.methods,
        output_path=output_path,
        overwrite=args.overwrite,
    )

    print("=============================================")
    print("Robustness Experiment Configuration")
    print("=============================================")
    print(cfg.to_json())
    print("=============================================\n", flush=True)

    scenarios = get_scenarios()

    tasks: List[Tuple[str, List[DriftPhase], str, int, int]] = []
    for scenario_name, phases in scenarios.items():
        for method in cfg.methods:
            for seed_idx in range(cfg.num_seeds):
                tasks.append((scenario_name, phases, method, cfg.num_gpus, seed_idx))

    total_runs = len(tasks)
    print(f"Total runs (scenario × method × seed): {total_runs}")
    n_procs = min(total_runs, cpu_count())
    print(f"Using {n_procs} worker processes.\n", flush=True)

    rows: List[Dict[str, float]] = []
    with Pool(processes=n_procs) as pool:
        for i, row in enumerate(pool.imap_unordered(_worker, tasks), 1):
            rows.append(row)
            # Progress every single completed run
            print(f"[{i:3d}/{total_runs}] robustness runs completed", flush=True)

    df = pd.DataFrame(rows)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_path, index=False)

    print("=============================================")
    print("Done.")
    print(f"Saved robustness metrics to: {cfg.output_path}")
    print("Rows:", len(df))
    print("=============================================", flush=True)


if __name__ == "__main__":
    main()

