# LSHFed: Extension Experiments

This folder contains extension experiments for *LSHFed (Robust and Communication-Efficient Federated Learning with Locally-Sensitive Hashing Gradient Mapping*, Guanjie Cheng et al., AAAI-26). 

LSHFed uses random hyperplane projections (LSHGM) to turn high-dimensional gradient updates into binary hashes. It then uses Hamming distance instead of Euclidean distance to make gradient verification faster and cheaper in robust federated learning (FL). 

The original paper uses a fixed number of hyperplane projections ($r$) and assumes Hamming distance is a good stand-in for Euclidean distance. However, it never tests how varying $r$ actually changes things. In practice, $r$ is a tradeoff: higher values might give better distance estimates, but generating those extra projections costs time and memory.

These experiments systematically test how $r$ impacts distance accuracy, model selection decisions, and compute overhead across four stages.

---

## Experiments

### Experiment 01: Within-Model Gradient Correlation
This experiment tests whether Hamming distance tracks Euclidean distance under normal conditions. We took 100 CIFAR-10 gradients from a ResNet-9 model at a fixed state, computed all pairwise distances, and swept $r$ from 1 to 256. Surprisingly, the correlation stayed low across the entire range. Even though adding more hyperplanes slightly raised the correlation, it was not high enough to make the Hamming distance match Euclidean distance on raw model gradients.

### Experiment 02: Synthetic Gradient Correlation
To figure out why Experiment 01 failed, we built synthetic gradients with controlled changes in magnitude vs. direction. Sweeping $r$ from 1 to 256 on these vectors showed that Hamming distance ignores magnitude scaling entirely, but correlates heavily with directional changes. Increasing $r$ consistently imrpoved the correlation for direction-perturbed gradients. This proves LSHGM is purely an angular/directional metric, explaining when and why it diverges from Euclidean distance.

### Experiment 03: Verifier Update-Distance Correlation
Next, we tested if this metric difference actually breaks real FL decisions. We ran an 8-round CIFAR-10 simulation with 5 local trainers and 2 aggregators using LSHFed's verification workflow. For every update pair, we calculated both distances and checked winner agreement, i.e., whether both metrics picked the same winning aggregator. We found that low distance correlation doesn't always cause bad decisions; the two metrics still selected the same aggregator most of the time.

### Experiment 04: Projection Count Tradeoff
Finally, we measured the full cost of $r$ on the 14 update pairs from Experiment 03. We swept $r$ while tracking distance correlation, selection accuracy, hash size, and runtime. Crucially, we timed the full LSH pipeline: projection matrix multiplication, hashing, and Hamming comparison. While comparing Hamming hashes is almost instant, matrix projection takes real compute time that grows with $r$.

---

## Overall Investigation

The four experiments follow a simple progression: 
* gradient distances 
* controlled gradient changes 
* actual FL updates 
* r and cost tradeoff. 

The aim is to better understand the behavior of LSHGM when r is changed, especially whether a larger projection count actually improves the decisions made from Hamming distance and whether any improvement is worth the additional computation and hash size.