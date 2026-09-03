# Experiment 01: Within-Model Gradient Correlation

## 1. Motivation / Objective

LSHFed uses random-hyperplane locality-sensitive hashing (LSH) to represent model gradients as binary hashes. The distance between two gradients can then be estimated using the Hamming distance between their hashes instead of directly comparing the full gradient tensors.

This experiment starts with a basic question: how well does this Hamming distance track the actual Euclidean distance between gradients at different values of \(r\)?

Before looking at a complete federated-learning trajectory, we first test the relationship in a simpler, less computationally-expensive setting. A single model is kept fixed while multiple stochastic gradients are generated from it. This removes changes in the model state and lets us focus on the relationship between the gradients themselves.

We also sweep the number of random hyperplanes, \(r\). If the low correlation were mainly caused by using too few projections, increasing \(r\) should improve the relationship.

The experiment therefore asks:

* How strongly are Euclidean and LSH Hamming distances correlated for gradients from the same model?
* Does increasing \(r\) substantially improve that correlation?

---

## 2. Methodology

### Experimental setup

The experiment uses the CIFAR-10 dataset and a ResNet9-style model. The model is kept at one fixed state while stochastic gradients are collected.

For each of 100 gradient samples, the same model parameters are used, while the training data and stochastic behavior provide the variation between gradients.

The gradient used for the experiment is the model parameter tensor with shape

$$
512 \times 512 \times 3 \times 3.
$$

This gives 100 gradients and therefore

$$
\binom{100}{2}=4,950
$$

unique gradient pairs.

### Distance comparison

For two gradients \(g_i\) and \(g_j\), the reference distance is their Euclidean distance:

$$
d_E(g_i,g_j)=\|g_i-g_j\|_2.
$$

The LSHFed hashing procedure first flattens the gradient using the same 2D representation used by the authors' implementation. Random projection vectors are then applied and the sign of each projection produces a binary hash.

The Hamming distance between two hashes is used as the LSH-based distance:

$$
d_H(h_i,h_j)=\sum_k \mathbf{1}[h_{i,k}\neq h_{j,k}].
$$

The experiment compares these two distances over all 4,950 gradient pairs.

### Projection sweep

The tested projection counts are

$$
r\in\{1,2,4,5,8,16,32,64,128,256\}.
$$

A fixed bank of 256 random projection vectors is generated, and smaller values of \(r\) use prefixes of the same bank. This keeps the projection randomness fixed while \(r\) changes.

For every \(r\), Pearson and Spearman correlations are calculated between the Euclidean and Hamming distances.

---

## 3. Results and Interpretation

The Euclidean–Hamming relationship is weak across the entire sweep.

At \(r=1\), the Pearson correlation is only about 0.014. Even at \(r=256\), it reaches only about 0.131. Spearman correlation follows the same pattern, reaching only about 0.126 at \(r=256\). 

Even though the correlation does increase from 0.014 to 0.131 due to increaing \(r\), the highest correlation reached at \(r=256\) is still very low.

These results motivate Experiment 02, where gradient differences are constructed synthetically instead of being taken from the same model, so that the effect of magnitude and directional changes on Euclidean and Hamming distances can be examined.

---

## 4. Outputs

The `results/` directory contains the numerical results and figures produced by the notebook.
