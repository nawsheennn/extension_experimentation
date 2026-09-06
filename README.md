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

### Reproducibility

All four experiments of LSHFed are self-contained. They can be downloaded and ran anywhere.

---

## GGSS-R Extension

The second extension focuses on GGSS-R from the *Enhanced Privacy Leakage from Noise-Perturbed Gradients via Gradient-Guided Conditional Diffusion Models* paper by Jiayang Meng et al. It studies the attack side of federated learning privacy, using gradient-guided diffusion sampling to reconstruct private client images from gradients exposed during FL. The experiments here investigate two aspects left underexplored in the original work: how victim architecture and training state affect gradient identifiability and reconstruction vulnerability, and whether dynamic guidance can improve reconstruction from perturbed gradients.

The investigation progresses through nine experiments:

1. *Baseline reconstruction:* Reproduces the GGSS-R attack with a locally trained single-epoch victim and establishes the initial reconstruction failure.
2. *Training-member reconstruction:* Tests whether longer victim training and using a known training-set target improve reconstruction; instead, reconstruction deteriorates further.
3. *Unperturbed reconstruction diagnostics:* Uses the authors' exact diffusion checkpoint and adds detailed diagnostics to isolate where the failed reconstruction originates.
4. *Unconditional diffusion diagnosis:* Verifies that the referenced diffusion model independently generates valid natural face images from noise.
5. *GSS without gradient guidance:* Tests the active GSS sampling path without leaked-gradient guidance, confirming that it also remains on the natural image manifold.
6. *Direct gradient inversion:* Removes diffusion entirely and shows that strong gradient matching can still fail to recover the target image.
7. *Gradient identifiability analysis:* Examines population- and layer-level gradients to explain why some victim states provide weak image-specific reconstruction signals.
8. *Victim architecture and state comparison:* Compares four architectures at initialization and after training, showing that reconstruction vulnerability depends on both architecture and training state through their effect on gradient propagation and identifiability.
9. *Scheduled guidance under gradient perturbation:* Extends GGSS-R's fixed guidance rate with linear decay and finds better reconstruction from noisy gradients by reducing gradient influence during later denoising.

For full implementation details, diagnostic results, and experiment-level findings, see the [`GGSSR/`](./GGSSR/) directory.

### Reproducibility

Experiments 01 (baseline reconstruction), 02 (training-member reconstruction), 03 (unperturbed reconstruction diagnostics), and 09 (scheduled guidance under gradient perturbation) are self-contained experiments. Each of their notebooks can be downloaded and ran anywhere independently.

Experiments 04 (unconditional diffusion), 05 (gss without gradient guidance), 06 (direct gradient inversion), 07 (gradient identifiability analyses) and 08 (victim architecture and state comparison) use files from my personal Drive as input. They are thus not self-contained, and cannot be executed independently.