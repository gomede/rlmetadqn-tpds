#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_simulated_experiments_tpds.py

TPDS-grade simulation driver.

- Trains RLMetaDQN as a meta-scheduler across:
    * 4 workload shapes: base, bursty, diurnal, shifted_diurnal
    * 3 intensity levels: low / medium / high
    * 4 GPU counts: 2, 4, 8, 16

- Evaluates:
    * StaticEqual
    * StaticPriority
    * EDF
    * SRPT
    * DynamicHeuristic
    * RLMetaDQN (meta-scheduler choosing among the above)

- Outputs:
    * rl_meta_dqn_tpds.pt            (trained policy)
    * tpds_dqn_training_rewards.csv  (learning curve)
    * tpds_sim_metrics.csv           (per-run metrics for analysis)
"""

import math
import random
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Scheduler names (must match real-system script)
# ----------------------------------------------------------------------

SCHEDULER_NAMES = [
    "StaticEqual",
    "StaticPriority",
    "EDF",
    "SRPT",
    "DynamicHeuristic",
]


# ----------------------------------------------------------------------
# Small ASCII progress bar helper (for nested progress feeling)
# ----------------------------------------------------------------------

def format_bar(current: int, total: int, width: int = 20) -> str:
    """ASCII progress bar: [#####-----] current/total"""
    if total <= 0:
        total = 1
    frac = current / total
    filled = int(width * frac)
    empty = width - filled
    return "[" + "#" * filled + "-" * empty + f"] {current}/{total}"


# ======================================================================
# SLA-aware reward (same as in real-system script)
# ======================================================================

@dataclass
class RewardConfig:
    max_slowdown: float = 10.0
    max_wait_time: float = 3600.0
    max_queue: int = 100
    max_power_w: float = 800.0

    lambda_slow: float = 1.0
    lambda_sla: float = 5.0
    lambda_wait: float = 0.5

    lambda_util: float = 0.1
    lambda_queue: float = 0.1
    lambda_energy: float = 0.05

    clip_min: float = -10.0
    clip_max: float = 2.0


@dataclass
class FinishedJobInfo:
    submit_time: float
    start_time: float
    finish_time: float
    est_runtime: float
    sla_missed: bool


def compute_rl_reward(
    finished_jobs: List[FinishedJobInfo],
    busy_gpus: int,
    num_gpus: int,
    queue_len: int,
    total_power_w: Optional[float],
    cfg: RewardConfig,
) -> float:
    reward = 0.0

    for job in finished_jobs:
        runtime = max(job.est_runtime, 1e-3)
        flow_time = job.finish_time - job.submit_time
        slowdown = flow_time / runtime
        slowdown_norm = min(slowdown, cfg.max_slowdown) / cfg.max_slowdown

        wait_time = job.start_time - job.submit_time
        wait_norm = min(wait_time, cfg.max_wait_time) / cfg.max_wait_time

        sla_penalty = 1.0 if job.sla_missed else 0.0

        r_finish = (
            - cfg.lambda_slow * slowdown_norm
            - cfg.lambda_sla * sla_penalty
            - cfg.lambda_wait * wait_norm
        )
        reward += r_finish

    util = (busy_gpus / num_gpus) if num_gpus > 0 else 0.0
    util_term = cfg.lambda_util * util

    queue_norm = min(queue_len, cfg.max_queue) / max(cfg.max_queue, 1)
    queue_term = cfg.lambda_queue * queue_norm

    if total_power_w is not None and cfg.max_power_w > 0:
        power_norm = min(total_power_w, cfg.max_power_w) / cfg.max_power_w
    else:
        power_norm = 0.0
    energy_term = cfg.lambda_energy * power_norm

    reward += util_term
    reward -= queue_term
    reward -= energy_term

    reward = max(cfg.clip_min, min(cfg.clip_max, reward))
    return reward


# ======================================================================
# Simulation job model
# ======================================================================

@dataclass
class SimJob:
    job_id: int
    submit_time: float
    est_runtime_s: float
    deadline_s: float
    priority: int

    start_time: Optional[float] = None
    end_time: Optional[float] = None
    remaining_s: float = 0.0
    sla_violated: bool = False

    def reset_runtime(self):
        self.remaining_s = self.est_runtime_s

    @property
    def completion_time(self) -> Optional[float]:
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def slowdown(self) -> Optional[float]:
        if self.completion_time is None or self.est_runtime_s <= 0:
            return None
        return self.completion_time / self.est_runtime_s


@dataclass
class SimGpu:
    gpu_id: int
    job: Optional[SimJob] = None


# ======================================================================
# DQN network (must match real-system script)
# ======================================================================

class DQNNet(nn.Module):
    def __init__(self, state_dim: int = 9, n_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def encode_state(workload_label: str,
                 intensity_label: str,
                 num_gpus: int,
                 num_jobs: int) -> np.ndarray:
    w = np.zeros(4, dtype=np.float32)
    if workload_label == "base":
        w[0] = 1.0
    elif workload_label == "bursty":
        w[1] = 1.0
    elif workload_label == "diurnal":
        w[2] = 1.0
    elif workload_label == "shifted_diurnal":
        w[3] = 1.0

    i = np.zeros(3, dtype=np.float32)
    if intensity_label == "low":
        i[0] = 1.0
    elif intensity_label == "medium":
        i[1] = 1.0
    elif intensity_label == "high":
        i[2] = 1.0

    g = (num_gpus - 2) / 14.0
    n = np.tanh(num_jobs / 50.0)
    return np.concatenate([w, i, np.array([g, n], dtype=np.float32)], axis=0)


# ======================================================================
# Heuristic schedulers (same semantics as real-system version)
# ======================================================================

def pick_job_heuristic(pending: List[SimJob],
                       policy: str,
                       now: float) -> int:
    if not pending:
        raise ValueError("No pending jobs")

    if policy == "SRPT":
        key_fn = lambda j: j.est_runtime_s
    elif policy == "EDF":
        key_fn = lambda j: j.submit_time + j.deadline_s
    elif policy == "StaticPriority":
        key_fn = lambda j: (-j.priority, j.submit_time)
    elif policy == "StaticEqual":
        key_fn = lambda j: j.submit_time
    elif policy == "DynamicHeuristic":
        def dyn_key(j: SimJob):
            deadline_abs = j.submit_time + j.deadline_s
            slack = max(deadline_abs - now, 1.0)
            return slack / (j.priority + 1.0)
        key_fn = dyn_key
    else:
        raise ValueError(f"Unknown policy: {policy}")

    best_idx = min(range(len(pending)), key=lambda idx: key_fn(pending[idx]))
    return best_idx


# ======================================================================
# Simple synthetic workload generator
# ======================================================================

def generate_jobs(
    workload_label: str,
    intensity_label: str,
    lam_base: float,
    seed: int,
    horizon_s: float = 3600.0,
) -> List[SimJob]:
    """
    Generate a Poisson-like arrival process with different shapes.

    workload_label:
        base            -> homogeneous Poisson
        bursty          -> alternating high/low lambda
        diurnal         -> sinusoidal lambda
        shifted_diurnal -> phase-shifted sinusoidal lambda

    intensity_label:
        low / medium / high -> scales lambda.
    """
    rng = random.Random(seed)

    if intensity_label == "low":
        lam = lam_base
    elif intensity_label == "medium":
        lam = 2.0 * lam_base
    else:  # high
        lam = 4.0 * lam_base

    jobs: List[SimJob] = []
    t = 0.0
    jid = 0

    def current_lambda(t_now: float) -> float:
        if workload_label == "base":
            return lam
        elif workload_label == "bursty":
            # alternate every 300s
            period = 300.0
            phase = int(t_now // period)
            return lam * (3.0 if phase % 2 == 0 else 0.3)
        elif workload_label == "diurnal":
            # 1h horizon normalized to [0, 2π]
            x = 2 * math.pi * (t_now / horizon_s)
            return lam * (0.5 + 0.5 * (1 + math.sin(x)))
        elif workload_label == "shifted_diurnal":
            x = 2 * math.pi * ((t_now + horizon_s / 4) / horizon_s)
            return lam * (0.5 + 0.5 * (1 + math.sin(x)))
        else:
            return lam

    while t < horizon_s:
        lam_t = max(current_lambda(t), 1e-6)
        dt = rng.expovariate(lam_t)
        t += dt
        if t >= horizon_s:
            break

        # random runtime and deadline
        est_runtime_s = rng.uniform(30.0, 900.0)  # 0.5–15 minutes
        deadline_factor = rng.uniform(1.2, 2.0)
        deadline_s = est_runtime_s * deadline_factor
        priority = rng.randint(0, 3)

        jobs.append(
            SimJob(
                job_id=jid,
                submit_time=t,
                est_runtime_s=est_runtime_s,
                deadline_s=deadline_s,
                priority=priority,
                remaining_s=est_runtime_s,
            )
        )
        jid += 1

    return jobs


# ======================================================================
# Simulation environment
# ======================================================================

@dataclass
class SimConfig:
    num_gpus: int
    workload_label: str
    intensity_label: str
    lam_base: float
    horizon_s: float = 3600.0
    dt_s: float = 5.0
    power_idle_w: float = 100.0
    power_per_gpu_w: float = 250.0
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)


class SimEnv:
    def __init__(self, cfg: SimConfig, seed: int):
        self.cfg = cfg
        self.seed = seed
        self.rng = random.Random(seed)

        self.jobs_all: List[SimJob] = generate_jobs(
            workload_label=cfg.workload_label,
            intensity_label=cfg.intensity_label,
            lam_base=cfg.lam_base,
            seed=seed,
            horizon_s=cfg.horizon_s,
        )
        self.jobs_all.sort(key=lambda j: j.submit_time)
        for j in self.jobs_all:
            j.reset_runtime()

        self.time = 0.0
        self.pending: List[SimJob] = []
        self.future: List[SimJob] = list(self.jobs_all)
        self.completed: List[SimJob] = []

        self.gpus: List[SimGpu] = [SimGpu(gpu_id=i) for i in range(cfg.num_gpus)]
        self.done = False

    def _inject_arrivals(self):
        while self.future and self.future[0].submit_time <= self.time:
            j = self.future.pop(0)
            self.pending.append(j)

    def _simulate_dt(self, dt: float) -> List[FinishedJobInfo]:
        finished_this_step: List[FinishedJobInfo] = []

        # Run jobs on GPUs
        for g in self.gpus:
            if g.job is not None:
                g.job.remaining_s -= dt
                if g.job.remaining_s <= 0.0:
                    g.job.end_time = self.time
                    deadline_abs = g.job.submit_time + g.job.deadline_s
                    g.job.sla_violated = g.job.end_time > deadline_abs
                    self.completed.append(g.job)
                    finished_this_step.append(
                        FinishedJobInfo(
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

    def step(self, policy: str) -> Tuple[np.ndarray, float, bool]:
        """
        Single RL step: we assume 1 scheduler decision per dt.
        """
        cfg = self.cfg

        # Advance time by dt, injecting new jobs
        self._inject_arrivals()
        finished_before = self._simulate_dt(cfg.dt_s)

        # After running dt with old schedule, we make a new decision & schedule
        busy_gpus = sum(1 for g in self.gpus if g.job is not None)
        queue_len = len(self.pending)
        power_w = self._power_draw()

        reward = compute_rl_reward(
            finished_jobs=finished_before,
            busy_gpus=busy_gpus,
            num_gpus=cfg.num_gpus,
            queue_len=queue_len,
            total_power_w=power_w,
            cfg=cfg.reward_cfg,
        )

        # Now apply new policy to schedule any idle GPUs
        self._schedule_jobs(policy)

        # Move time forward
        self.time += cfg.dt_s
        self._inject_arrivals()

        # termination condition
        if self.time >= cfg.horizon_s and not self.pending and \
                all(g.job is None for g in self.gpus):
            self.done = True

        state = encode_state(
            workload_label=cfg.workload_label,
            intensity_label=cfg.intensity_label,
            num_gpus=cfg.num_gpus,
            num_jobs=len(self.pending),
        )
        return state, reward, self.done

    def reset_state(self) -> np.ndarray:
        self.time = 0.0
        self.pending = []
        self.future = list(self.jobs_all)
        self.completed = []
        for g in self.gpus:
            g.job = None
        self.done = False
        self._inject_arrivals()
        return encode_state(
            workload_label=self.cfg.workload_label,
            intensity_label=self.cfg.intensity_label,
            num_gpus=self.cfg.num_gpus,
            num_jobs=len(self.pending),
        )


# ======================================================================
# Replay buffer + DQN training
# ======================================================================

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s_next: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.idx = 0

    def push(self, tr: Transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(tr)
        else:
            self.buffer[self.idx] = tr
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        # stack
        s = np.stack([b.s for b in batch])
        s_next = np.stack([b.s_next for b in batch])
        a = np.array([b.a for b in batch], dtype=np.int64)
        r = np.array([b.r for b in batch], dtype=np.float32)
        d = np.array([b.done for b in batch], dtype=np.bool_)
        return Transition(s=s, a=a, r=r, s_next=s_next, done=d)

    def __len__(self):
        return len(self.buffer)


def train_dqn_on_env(
    device: torch.device,
    env_cfg: SimConfig,
    seed: int,
    episodes: int,
    gamma: float,
    lr: float,
    batch_size: int,
    replay_capacity: int,
) -> Tuple[DQNNet, List[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    state_dim = 9
    n_actions = len(SCHEDULER_NAMES)

    policy_net = DQNNet(state_dim=state_dim, n_actions=n_actions).to(device)
    target_net = DQNNet(state_dim=state_dim, n_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(replay_capacity)

    eps_start = 1.0
    eps_end = 0.05
    eps_decay = max(episodes // 2, 1)

    returns: List[float] = []

    print(f"[TRAIN] Starting RLMetaDQN training for {episodes} episodes")

    for ep in range(episodes):
        env = SimEnv(env_cfg, seed + ep)  # different randomness per episode
        s = env.reset_state()
        done = False
        ep_return = 0.0

        step_idx = 0
        eps = eps_start  # default in case loop ends immediately
        while not done:
            # Epsilon-greedy
            eps = eps_end + (eps_start - eps_end) * math.exp(-step_idx / eps_decay)
            if random.random() < eps:
                a = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    q_vals = policy_net(s_t)
                    a = int(torch.argmax(q_vals, dim=1).item())

            policy_name = SCHEDULER_NAMES[a]
            s_next, r, done = env.step(policy_name)

            buffer.push(Transition(s=s, a=a, r=r, s_next=s_next, done=done))
            ep_return += r
            s = s_next
            step_idx += 1

            # DQN update
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                bs = torch.tensor(batch.s, dtype=torch.float32, device=device)
                ba = torch.tensor(batch.a, dtype=torch.long, device=device)
                br = torch.tensor(batch.r, dtype=torch.float32, device=device)
                bs_next = torch.tensor(batch.s_next, dtype=torch.float32, device=device)
                bdone = torch.tensor(batch.done, dtype=torch.float32, device=device)

                q_pred = policy_net(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(bs_next).max(1)[0]
                    q_target = br + gamma * q_next * (1.0 - bdone)

                loss = F.mse_loss(q_pred, q_target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        # soft target update
        tau = 0.01
        for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
            tp.data.copy_(tau * pp.data + (1.0 - tau) * tp.data)

        returns.append(ep_return)

        # Progress bar over episodes (nested feel)
        bar = format_bar(ep + 1, episodes, width=30)
        print(f"[TRAIN] {bar} ep={ep} return={ep_return:.3f} eps={eps:.3f}")

    return policy_net, returns


# ======================================================================
# Evaluation on simulated envs (metrics CSV)
# ======================================================================

def eval_scheduler_on_env(
    env_cfg: SimConfig,
    scheduler: str,
    seed: int,
    episodes: int,
) -> Dict[str, float]:
    """
    Evaluate a FIXED scheduler on env_cfg over multiple episodes, returning
    averaged metrics.
    """
    all_jcts = []
    all_slow = []
    all_p95_jct = []
    all_p95_slow = []
    all_sla = []
    all_util = []
    all_energy = []

    for ep in range(episodes):
        env = SimEnv(env_cfg, seed + ep)
        cfg = env.cfg
        reward_cfg = cfg.reward_cfg

        total_return = 0.0
        util_sum = 0.0
        util_samples = 0
        energy_kwh = 0.0

        state = env.reset_state()
        done = False

        while not done:
            finished_before = env._simulate_dt(cfg.dt_s)
            busy_gpus = sum(1 for g in env.gpus if g.job is not None)
            queue_len = len(env.pending)
            power_w = env._power_draw()

            r = compute_rl_reward(
                finished_jobs=finished_before,
                busy_gpus=busy_gpus,
                num_gpus=cfg.num_gpus,
                queue_len=queue_len,
                total_power_w=power_w,
                cfg=reward_cfg,
            )
            total_return += r

            # schedule with fixed scheduler
            env._schedule_jobs(scheduler)

            # bookkeeping
            util_sum += busy_gpus / cfg.num_gpus
            util_samples += 1
            energy_kwh += (power_w * (cfg.dt_s / 3600.0)) / 1000.0

            env.time += cfg.dt_s
            env._inject_arrivals()

            if env.time >= cfg.horizon_s and not env.pending and \
                    all(g.job is None for g in env.gpus):
                env.done = True

            done = env.done

        # collect metrics
        jcts = [j.completion_time for j in env.completed if j.completion_time is not None]
        slows = [j.slowdown for j in env.completed if j.slowdown is not None]
        if jcts:
            all_jcts.append(float(np.mean(jcts)))
            all_p95_jct.append(float(np.percentile(jcts, 95)))
        if slows:
            all_slow.append(float(np.mean(slows)))
            all_p95_slow.append(float(np.percentile(slows, 95)))

        sla_rate = np.mean([1.0 if j.sla_violated else 0.0 for j in env.completed])
        all_sla.append(float(sla_rate))

        avg_util = util_sum / max(util_samples, 1)
        all_util.append(float(avg_util * 100.0))
        all_energy.append(float(energy_kwh))

    return {
        "avg_jct": float(np.mean(all_jcts)) if all_jcts else 0.0,
        "avg_slowdown": float(np.mean(all_slow)) if all_slow else 0.0,
        "p95_jct": float(np.mean(all_p95_jct)) if all_p95_jct else 0.0,
        "p95_slowdown": float(np.mean(all_p95_slow)) if all_p95_slow else 0.0,
        "sla_rate": float(np.mean(all_sla)) if all_sla else 0.0,
        "energy_kwh": float(np.mean(all_energy)) if all_energy else 0.0,
        "avg_util": float(np.mean(all_util)) if all_util else 0.0,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    workloads = ["base", "bursty", "diurnal", "shifted_diurnal"]
    intensities = ["low", "medium", "high"]
    gpu_list = [2, 4, 8, 16]

    n_workloads = len(workloads)
    n_intensities = len(intensities)
    n_gpus = len(gpu_list)
    total_configs = n_workloads * n_intensities * n_gpus

    base_lam = 1.0  # base arrival rate (per 60s)
    lam_base = base_lam / 60.0

    train_env_cfg = SimConfig(
        num_gpus=4,
        workload_label="base",
        intensity_label="medium",
        lam_base=lam_base,
        horizon_s=3600.0,
        dt_s=5.0,
        reward_cfg=RewardConfig(),
    )

    print(
        f"[SIM] Device={device}, training env: "
        f"workload=base, intensity=medium, GPUs=4"
    )

    # -------------------------------
    # 1) Train RLMetaDQN
    # -------------------------------
    policy_net, returns = train_dqn_on_env(
        device=device,
        env_cfg=train_env_cfg,
        seed=0,
        episodes=200,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        replay_capacity=5000,
    )

    torch.save(policy_net.state_dict(), "rl_meta_dqn_tpds.pt")
    print("[SAVE] rl_meta_dqn_tpds.pt")

    # log training returns
    with open("tpds_dqn_training_rewards.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "return"])
        for ep, ret in enumerate(returns):
            w.writerow([ep, ret])

    # -------------------------------
    # 2) Evaluate all schedulers
    # -------------------------------
    metrics_path = "tpds_sim_metrics.csv"

    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "workload",
            "intensity_label",
            "lam",
            "n_gpus",
            "scheduler",
            "seed",
            "sla_rate",
            "energy_kwh",
            "avg_util",
            "avg_jct",
            "avg_slowdown",
            "p95_jct",
            "p95_slowdown",
        ])

    seed = 123

    print(
        f"[SIM] Starting evaluation: "
        f"{n_workloads} workloads × {n_intensities} intensities × "
        f"{n_gpus} GPU configs"
    )

    global_cfg_idx = 0

    for w_idx, workload in enumerate(workloads, start=1):
        print()
        print(f"[WORK] {format_bar(w_idx, n_workloads)} workload={workload}")

        for i_idx, intensity in enumerate(intensities, start=1):
            print(f"  [INT ] {format_bar(i_idx, n_intensities)} intensity={intensity}")

            for g_idx, n_gpus in enumerate(gpu_list, start=1):
                global_cfg_idx += 1
                print(
                    f"    [CONF] {format_bar(global_cfg_idx, total_configs)} "
                    f"workload={workload}, intensity={intensity}, GPUs={n_gpus}",
                    flush=True,
                )

                env_cfg = SimConfig(
                    num_gpus=n_gpus,
                    workload_label=workload,
                    intensity_label=intensity,
                    lam_base=lam_base,
                    horizon_s=3600.0,
                    dt_s=5.0,
                    reward_cfg=RewardConfig(),
                )

                # Total schedulers for this config (heuristics + RLMetaDQN)
                sched_total = len(SCHEDULER_NAMES) + 1
                sched_idx = 0

                # heuristics
                for sched in SCHEDULER_NAMES:
                    sched_idx += 1
                    print(
                        f"      [SCHED] {format_bar(sched_idx, sched_total)} "
                        f"scheduler={sched}",
                        flush=True,
                    )
                    res = eval_scheduler_on_env(
                        env_cfg=env_cfg,
                        scheduler=sched,
                        seed=seed,
                        episodes=3,
                    )
                    with open(metrics_path, "a", newline="") as f:
                        w = csv.writer(f)
                        w.writerow([
                            workload,
                            intensity,
                            lam_base,
                            n_gpus,
                            sched,
                            seed,
                            res["sla_rate"],
                            res["energy_kwh"],
                            res["avg_util"],
                            res["avg_jct"],
                            res["avg_slowdown"],
                            res["p95_jct"],
                            res["p95_slowdown"],
                        ])

                # RLMetaDQN evaluation: treat it as scheduler name "RLMetaDQN"
                sched_idx += 1
                print(
                    f"      [SCHED] {format_bar(sched_idx, sched_total)} "
                    f"scheduler=RLMetaDQN",
                    flush=True,
                )

                env = SimEnv(env_cfg, seed)
                cfg = env_cfg
                reward_cfg = cfg.reward_cfg
                total_return_all = []
                all_jcts = []
                all_slow = []
                all_p95_jct = []
                all_p95_slow = []
                all_sla = []
                all_util = []
                all_energy = []

                for ep in range(3):
                    env = SimEnv(env_cfg, seed + ep)
                    s = env.reset_state()
                    done = False
                    total_ret = 0.0
                    util_sum = 0.0
                    util_samples = 0
                    energy_kwh = 0.0

                    while not done:
                        with torch.no_grad():
                            s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                            q_vals = policy_net(s_t)
                            a = int(torch.argmax(q_vals, dim=1).item())
                        policy_name = SCHEDULER_NAMES[a]

                        finished_before = env._simulate_dt(cfg.dt_s)
                        busy_gpus = sum(1 for g in env.gpus if g.job is not None)
                        queue_len = len(env.pending)
                        power_w = env._power_draw()

                        r = compute_rl_reward(
                            finished_jobs=finished_before,
                            busy_gpus=busy_gpus,
                            num_gpus=cfg.num_gpus,
                            queue_len=queue_len,
                            total_power_w=power_w,
                            cfg=reward_cfg,
                        )
                        total_ret += r

                        env._schedule_jobs(policy_name)

                        util_sum += busy_gpus / cfg.num_gpus
                        util_samples += 1
                        energy_kwh += (power_w * (cfg.dt_s / 3600.0)) / 1000.0

                        env.time += cfg.dt_s
                        env._inject_arrivals()

                        if env.time >= cfg.horizon_s and not env.pending and \
                                all(g.job is None for g in env.gpus):
                            env.done = True
                        done = env.done

                        s = encode_state(
                            workload_label=cfg.workload_label,
                            intensity_label=cfg.intensity_label,
                            num_gpus=cfg.num_gpus,
                            num_jobs=len(env.pending),
                        )

                    total_return_all.append(total_ret)

                    jcts = [j.completion_time for j in env.completed if j.completion_time is not None]
                    slows = [j.slowdown for j in env.completed if j.slowdown is not None]
                    if jcts:
                        all_jcts.append(float(np.mean(jcts)))
                        all_p95_jct.append(float(np.percentile(jcts, 95)))
                    if slows:
                        all_slow.append(float(np.mean(slows)))
                        all_p95_slow.append(float(np.percentile(slows, 95)))

                    sla_rate = np.mean([1.0 if j.sla_violated else 0.0 for j in env.completed])
                    all_sla.append(float(sla_rate))

                    avg_util = util_sum / max(util_samples, 1)
                    all_util.append(float(avg_util * 100.0))
                    all_energy.append(float(energy_kwh))

                res_rl = {
                    "avg_jct": float(np.mean(all_jcts)) if all_jcts else 0.0,
                    "avg_slowdown": float(np.mean(all_slow)) if all_slow else 0.0,
                    "p95_jct": float(np.mean(all_p95_jct)) if all_p95_jct else 0.0,
                    "p95_slowdown": float(np.mean(all_p95_slow)) if all_p95_slow else 0.0,
                    "sla_rate": float(np.mean(all_sla)) if all_sla else 0.0,
                    "energy_kwh": float(np.mean(all_energy)) if all_energy else 0.0,
                    "avg_util": float(np.mean(all_util)) if all_util else 0.0,
                }

                with open(metrics_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        workload,
                        intensity,
                        lam_base,
                        n_gpus,
                        "RLMetaDQN",
                        seed,
                        res_rl["sla_rate"],
                        res_rl["energy_kwh"],
                        res_rl["avg_util"],
                        res_rl["avg_jct"],
                        res_rl["avg_slowdown"],
                        res_rl["p95_jct"],
                        res_rl["p95_slowdown"],
                    ])

    print("All simulated experiments completed.")


if __name__ == "__main__":
    main()

