# Experiment 04: Projection Count Tradeoff

## 1. Motivation / Objective

The previous experiment 03 moved from synthetic gradients to an actual federated-learning trajectory. It found weak overall Hamming–Euclidean correlation, but the two metrics still selected the same aggregator in 6 of 7 rounds.

This raises a practical question: how does the choice of \(r\) affect the actual aggregator-selection decision?

A larger number of random hyperplanes gives a longer hash, which can provide a better representational capacity and a more granular resolution. This may change how closely the Hamming distance follows Euclidean distance. However, it requires more projection operations. Therefore, more projections have both a potential benefit and a cost.

Experiment 04 studies this tradeoff by sweeping

$$
r\in\{1,2,4,5,8,16,32,64,128,256\}.
$$

For each value of \(r\), we measure:

* correlation (Pearson and Spearman),
* winner agreement with Euclidean distance,
* hash-generation time,
* Hamming-comparison time,
* hash representation size.

The main goal is not to assume that a larger \(r\) is better, but to see what actually happens on the federated trajectory.

---

## 2. Methodology

### Fixed federated trajectory

Running a new federated trajectory for every \(r\) would change more than the projection count. A different \(r\) could lead to a different aggregator choice, which would then produce different model updates in later rounds.

To isolate the effect of \(r\), this experiment first generates one fixed federated trajectory using the baseline \(r=5\).

The candidate/benchmark update pairs from all 8 rounds are then saved and frozen. Every value of \(r\) is evaluated on those same pairs.

This gives the following pipeline:

1. Generate one LSHFed-style trajectory with r=5
2. Record candidate/benchmark updates
3. Freeze pairs
4. Sweep r = 1, 2, 4, ... , 256 on the same frozen updates
5. LSH hashing + Hamming distance
6. Winner agreement + correlation
7. Runtime + hash size

The federated setup follows Experiment 03:

* 10,000 CIFAR-10 samples
* 5 Local Trainers
* 2 Aggregators
* 2,000 samples per Local Trainer
* 1 local epoch
* 8 rounds
* 14 candidate/benchmark comparisons

The complete model state is still passed between rounds, while the distance calculations use the trainable parameter updates.

### LSH projection banks

A master bank of 256 standard-normal projection vectors is generated for each required vector dimension. Each smaller \(r\) uses a prefix of the corresponding bank.

This keeps the projection vectors consistent across the sweep.

The hashing implementation retains the per-vector `torch.mv` loop used by the authors' implementation. This is intentional because runtime is one of the quantities being measured.

### Hash size

Each additional projection adds another set of binary hash values. Therefore the representation size grows linearly with \(r\).

For example:

* \(r=5\): 80,580 bits, about 10 KB
* \(r=256\): 4,125,696 bits, about 516 KB

---

## 3. Results and Interpretation

### Winner agreement is not monotonic with \(r\)

The winner-agreement results are:

| \(r\) | Winner agreement |
| ----: | ---------------: |
|     1 |            85.7% |
|     2 |            85.7% |
|     4 |             100% |
|     5 |             100% |
|     8 |             100% |
|    16 |             100% |
|    32 |             100% |
|    64 |            85.7% |
|   128 |            71.4% |
|   256 |            71.4% |

In this trajectory, \(r=4\) through \(r=32\) all produce 100% winner agreement with the Euclidean reference.

Increasing \(r\) beyond this range does not improve the decision result. In fact, agreement falls to 85.7% at \(r=64\) and 71.4% at \(r=128\) and \(r=256\).

So the experiment does not show a simple relationship where more projections always produce better aggregator decisions.

### Correlation also changes non-monotonically

Pearson correlation varies substantially across the sweep:

| \(r\) | Pearson | Spearman |
| ----: | ------: | -------: |
|     1 |  0.0311 |   0.0725 |
|     2 |  0.2951 |   0.5604 |
|     4 |  0.4797 |   0.3890 |
|     5 |  0.6631 |   0.3978 |
|     8 |  0.5426 |   0.3319 |
|    16 |  0.3474 |   0.2308 |
|    32 |  0.5844 |   0.3934 |
|    64 |  0.6845 |   0.0681 |
|   128 |  0.7099 |   0.0769 |
|   256 |  0.6560 |   0.0945 |

The highest Pearson correlation occurs at \(r=128\), but that value does not give the highest winner agreement. This is another example of why distance correlation and aggregator-selection agreement should be treated as separate measurements.

### Increasing \(r\) has a clear computational cost

Runtime increases strongly with the number of projections.

Average total time for hash generation plus Hamming comparison per pair is approximately:

| \(r\) | Total time (s) |
| ----: | -------------: |
|     1 |         0.0139 |
|     5 |         0.0527 |
|    32 |         0.3034 |
|    64 |         0.6939 |
|   128 |         1.3476 |
|   256 |         2.6662 |

The hash representation grows in the same direction, from roughly 10 KB at \(r=5\) to roughly 516 KB at \(r=256\).

### Overall observation

For this controlled federated trajectory, the useful range is not simply "the largest possible \(r\)." Moderate values from \(r=4\) to \(r=32\) achieve perfect winner agreement in this run, while larger values substantially increase computation and hash size without improving that decision metric.

The results should be viewed as a measurement of the tradeoff on this particular trajectory. There are only 14 frozen candidate/benchmark comparisons, so the experiment does not establish a universal best value of \(r\).

What it does show is that increasing the projection count has a real cost, while the decision benefit is not guaranteed to increase with it.

---

## 4. Outputs

The `results/` directory contains the sweep results, including the correlation and winner-agreement tables, runtime measurements, hash-size calculations, and the corresponding plots.
