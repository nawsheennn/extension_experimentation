# Federated Learning: Paper Extensions

This repository contains implementations of extensions addressing gaps and open questions I identified in federated learning papers, with a focus on attack and defense mechanisms. The goal is to investigate specific limitations or unexamined design parameters through targeted experiments.

---

## Repository Structure

Each extension experiment lives in its own directory and includes:
* A Jupyter notebook containing code, implementation, and analysis
* An experiment-level `README.md` explaining the motivation and findings
* A `results/` folder (where applicable) containing generated outputs and figures

---

## LSHFed Extension

The first extension focuses on *LSHFed (Robust and Communication-Efficient Federated Learning with Locally-Sensitive Hashing Gradient Mapping*, Guanjie Cheng et al.). LSHFed targets Byzantine robustness in federated learning, using LSH-based gradient mapping (LSHGM) and Hamming distance comparisons to reduce the communication and computational cost of filtering malicious client updates.

This extension examines *$r$ (the number of hyperplane projections)* - a core hyperparameter that the original paper leaves largely unanalyzed. The experiments evaluate how changing $r$ affects distance fidelity, aggregator selection decisions, and total compute overhead across four progressive stages:

1. *Within-model gradients:* Evaluates Euclidean-Hamming correlation on gradients from a single model state while sweeping $r$.
2. *Synthetic gradients:* Isolates magnitude vs. directional shifts to identify when Hamming distance tracks or fails Euclidean distance.
3. *Federated learning updates:* Evaluates actual FL trajectories to test whether Euclidean and Hamming metrics select the same winning aggregator (*winner agreement*).
4. *Projection-count tradeoff:* Sweeps $r$ on live FL update pairs while benchmarking correlation, selection agreement, hash size, and end-to-end runtime (projection, hashing, and comparison).

For full implementation details, notebooks, and benchmarks, see the [`LSHFed/`](./LSHFed/) directory.