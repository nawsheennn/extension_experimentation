# Experiment 03: Verifier Update-Distance Correlation

## 1. Motivation / Objective

The first two experiments looked at the relationship between Euclidean and LSH Hamming distance in controlled settings.

Experiment 01 found low correlation for stochastic gradients from a fixed model. Experiment 02 showed that the relationship can be much stronger when gradient differences are generated through controlled directional perturbations.

The remaining question is what happens on actual updates produced by federated training.

This experiment therefore implements an LSHFed-style federated-learning trajectory with multiple local trainers and aggregators. Instead of only asking whether the two distances are correlated, it also asks the more practical question:

When Hamming distance is used to choose the aggregator, does Euclidean distance choose the same aggregator?

This distinction matters because LSHFed ultimately uses the distance comparison to make a selection. A weak global correlation does not necessarily mean that the selected aggregator will be different.

---

## 2. Methodology

### Federated setup

The experiment uses a 10,000-sample subset of CIFAR-10 with:

* 5 Local Trainers
* 2 Aggregators
* 2,000 samples per Local Trainer
* 1 local training epoch
* 8 federated rounds
* \(r=5\) for the LSH distance calculation

The aggregators are formed as:

$$
AG_0=\{LT_0,LT_2,LT_4\}
$$

and

$$
AG_1=\{LT_1,LT_3\}.
$$

Each round performs local training and aggregation, then compares the candidate aggregator updates with the current benchmark.

Round 0 has no previous benchmark, so AG0 is used to initialize it. Rounds 1 through 7 then provide two candidate-vs-benchmark comparisons per round, giving 14 distance observations in total.

### Federated trajectory

The pipeline is:

1. Local data 
2. Local training
3. Local Trainer updates
4. Aggregator updates
5. Compare candidate aggregators with benchmark
6. Select the Hamming-distance winner
7. Update the global model
8. Next round

The Hamming-based selection is the one used to continue the actual trajectory.

### Model state and distance calculation

The complete model state is passed between rounds so that the federated training process continues from the selected global model state.

The distance comparison itself is performed on the trainable parameter updates.

For two candidate updates \(U\) and \(V\), the Euclidean reference distance is

$$
d_E(U,V)=\sqrt{\sum_k\|U_k-V_k\|_2^2},
$$

where the sum is over the trainable parameter tensors.

The same updates are flattened and passed through the LSHFed-style random-hyperplane hashing procedure. Their Hamming distance is then

$$
d_H(H_U,H_V)=\sum_k\mathbf{1}[H_{U,k}\neq H_{V,k}].
$$

For every round, both distances are recorded for each candidate aggregator.

Two types of comparison are then made:

1. Pearson and Spearman correlation over all observations.
2. Winner agreement, checking whether Hamming and Euclidean choose the same aggregator within each round.

---

## 3. Results and Interpretation

The overall distance correlation is weak:

* Pearson = −0.1345
* Spearman = −0.4769

So the actual federated updates do not show the strong positive Hamming–Euclidean relationship observed in the controlled synthetic experiment.

However, the selection behavior tells a different story.

Hamming and Euclidean choose the same aggregator in 6 of the 7 rounds, giving:

$$
\text{Winner Agreement}=\frac{6}{7}=85.7\%.
$$

There is one clear disagreement. In round 4:

* Hamming distance selects AG1
* Euclidean distance selects AG0

One important observation is therefore that weak global distance correlation does not translate directly into frequent disagreement in the aggregator-selection decision.

Correlation measures how the two distances vary across all candidate/benchmark pairs. Winner agreement only asks which candidate is smaller within each round. Those are different properties.

This result changes the focus of the next experiment. Rather than looking only at correlation, Experiment 04 varies the number of projections \(r\) and measures winner agreement together with the computational and storage cost of the hash representation.
