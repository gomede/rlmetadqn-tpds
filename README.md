# RLMetaDQN: A Reinforcement Learning Meta-Scheduler for Adaptive Multi-GPU Cluster Workloads

This repository contains the simulation framework, real-system evaluation pipeline, and analysis scripts used in the paper:

**“A Reinforcement Learning Meta-Scheduler for Adaptive Multi-GPU Cluster Workloads”**  
*Submitted to IEEE Transactions on Parallel and Distributed Systems (TPDS), 2025.*

RLMetaDQN is a **meta–reinforcement learning scheduler** that selects among classical heuristics (SRPT, EDF, StaticPriority, StaticEqual, DynamicHeuristic) to deliver **robust, near-oracle performance** across heterogeneous, nonstationary GPU workloads.

It achieves:

- 1–2% oracle gap without regime labels  
- High robustness to noise, drift, and burst amplification  
- Zero-shot transfer from simulation to a real dual-A100 node  
- Superior fairness, tail latency, utilization, and queue stability compared to classical heuristics  

---

## 📁 Repository Structure

    .
    ├── real_traces/                    # Hardware-executed traces (10 × 50-job workloads)
    │
    ├── core_sim_metrics.csv            # 48-regime simulation results
    ├── ablation_metrics.csv            # Component elimination experiments
    ├── robustness_metrics.csv          # Stress tests and perturbation responses
    ├── real_system_metrics.csv         # Real A100-node performance logs
    │
    ├── run_core_sim.py                 # 3,072 core simulation runs across regimes
    ├── run_simulated_experiments_tpds.py
    │                                   # TPDS-grade simulation driver
    ├── run_real_system.py              # Real A100-node job executor
    ├── run_robustness.py               # Perturbation / drift robustness experiments
    ├── run_ablation_studies.py         # Component-level ablation framework
    │
    └── README.md

---

## 🚀 Quick Start

### 1. Clone the repository

    git clone https://github.com/<your-org>/rlmetadqn-tpds.git
    cd rlmetadqn-tpds

### 2. Create environment

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

> Simulation does **not** require GPUs. Hardware experiments do.

---

## 🧠 RLMetaDQN Overview

RLMetaDQN is **not** an end-to-end RL scheduler.  
Instead, it learns a compact policy that chooses *which heuristic to use* at each decision point:

    Action Space = { SRPT, EDF, StaticPriority, StaticEqual, DynamicHeuristic }

### Why this matters

End-to-end RL schedulers (DeepRM, DeepRM2, PPO, RLSchedule) often suffer from:

- Unstable training under nonstationary workloads  
- Large and variable action spaces  
- Poor simulation-to-hardware transfer  
- Collapse under drift and bursts  

Meta-scheduling retains **interpretability and robustness** while adding **adaptivity**: RLMetaDQN learns *when* to trust each analytic heuristic instead of replacing them.

---

## 📊 Experiments Included

This repository includes the complete experiment suite required to reproduce the TPDS paper.

---

### 1. Core Simulation (3,072 runs)

Simulates all workload shapes × intensities × GPU counts.

Run:

    python run_core_sim.py

Outputs:

- core_sim_metrics.csv  
- Utilization vs GPUs  
- Slowdown and JCT analysis  
- Tail-latency distributions  

---

### 2. Robustness Experiments (160 scenarios)

Perturbations include:

- Arrival noise (5–20%)  
- Duration distortions  
- Diurnal phase shifts  
- Burst amplification (1.5× to 3×)  
- GPU capacity dips  

Run:

    python run_robustness.py

Outputs:

- robustness_metrics.csv  
- Queue explosion detection  
- Seed variance analysis  

RLMetaDQN exhibits:

- Lowest performance degradation  
- 0 queue explosions  
- Smallest seed variance among tested schedulers  

---

### 3. Real-System A100 Experiments

Executes real GPU jobs (LLM fine-tuning, CNN/ViT training, ETL tasks) on a dual A100 node using the traces in `real_traces/`.

Run:

    python run_real_system.py --trace-dir real_traces/

Features:

- Actual GPU workloads using CUDA_VISIBLE_DEVICES  
- 1-second logging through nvidia-smi  
- Zero-shot transfer from simulation (no tuning)  
- Oracle-gap below 2% on hardware  

Outputs:

- real_system_metrics.csv  

---

### 4. Ablation Studies

Removes architectural elements to measure their contribution to performance and stability:

- Regime embeddings  
- Replay balancing  
- Deadline penalties  
- Energy term  
- Target network  
- Job-age features  

Run:

    python run_ablation_studies.py

Outputs:

- ablation_metrics.csv  

The ablations show that regime embeddings, deadline shaping, replay balancing, and target networks are especially important for maintaining robustness and queue stability.

---

## 📂 Dataset Description

### core_sim_metrics.csv

Per-run simulation metrics across 48 regimes (4 shapes × 3 intensities × 4 GPU counts). Columns include:

- workload_shape  
- intensity  
- gpu_count  
- scheduler  
- avg_jct, slowdown, makespan  
- queue_len_max, fairness_jain  
- utilization, energy  
- oracle_gap  

### robustness_metrics.csv

Perturbation experiment results, including:

- perturbation_type (noise, drift, burst, slowdown, etc.)  
- noise_level or severity  
- delta_slowdown vs clean regime  
- seed_variance  
- queue_explosions  

### ablation_metrics.csv

Each row corresponds to one ablated component, including:

- component_removed  
- delta_slowdown, delta_jct  
- delta_fairness  
- delta_queue_stability  

### real_system_metrics.csv

Real GPU execution metrics collected on the dual A100 node:

- Per-trace avg_jct, slowdown, fairness_jain  
- Power and energy estimates  
- GPU utilization (1-second sampling)  
- Meta-policy entropy  
- Oracle_gap per trace  

### real_traces/

Ten 50-job traces used in hardware experiments. These traces are generated from the same workload family as simulations but with held-out seeds and parameters, enabling clean evaluation of simulation-to-hardware transfer.

---

## 📐 Reproducing Figures

You can reproduce all the core figures from the TPDS paper, such as:

- util_vs_gpus.pdf (GPU utilization vs GPU count)  
- cdf_jct.pdf (CDF of job completion times)  
- oracle_gap.pdf (oracle-gap comparison)  
- robustness_radar.pdf (robustness across perturbations)  
- ablation_bar.pdf (component contribution bar plots)  

Example (scripts may vary depending on your local analysis setup):

    python analyze_core_sim.py
    python plot_util_vs_gpu.py

Check the repository’s analysis or plotting scripts (if present) to regenerate specific figures.

---

## ⚙️ Real-System Execution Details

RLMetaDQN is evaluated on real GPU workloads by wrapping actual training and inference commands. A typical job might be launched as:

    CUDA_VISIBLE_DEVICES=0 python job_script.py --runtime 120

The scheduler:

- Updates the cluster state at one-second granularity  
- Uses a lightweight MLP for heuristic selection  
- Logs GPU utilization, queue states, and job events  

Runtime overhead:

- Approximately 38–52 microseconds per meta-policy decision  
- Less than 0.005% of the 1-second scheduling interval  
- Independent of queue length due to the fixed-size action space  

This makes RLMetaDQN suitable for production-grade deployment in node-local schedulers.

---

## 📝 Citation

If you use this repository or build upon RLMetaDQN, please cite:

    Gomede, E., Palácios, R. H. C.
    "A Reinforcement Learning Meta-Scheduler for Adaptive Multi-GPU Cluster Workloads."
    IEEE Transactions on Parallel and Distributed Systems, 2025 (Submitted).

---

## 🧩 License

Please choose and add a license file (e.g., MIT, Apache 2.0, BSD) appropriate for your project and institutional policies.

---

## 🤝 Funding

This work **did not receive any specific grant** from funding agencies in the public, commercial, or not-for-profit sectors.

