# Experiment 02: Synthetic Gradient Correlation

## 1. Motivation / Objective

Experiment 01 found that Euclidean distance and LSH Hamming distance have very low correlation for stochastic gradients generated from the same model, and that increasing \(r\) results in a meager improvement of the correlation.

The next question is why.

The two distance measures do not respond to gradient differences in exactly the same way. Euclidean distance changes when either the magnitude or direction of a gradient changes. Random-hyperplane hashing, on the other hand, records the signs of projections and is therefore only sensitive to directional changes.

Experiment 02 uses synthetic gradients to isolate these effects.

It asks:

1. What happens when two gradients differ only in magnitude?
2. What happens when their direction is changed?
3. When gradient differences are generated through controlled directional perturbations, does increasing \(r\) improve Hamming–Euclidean correlation?

The purpose is not to reproduce the full LSHFed training process. It is to understand the geometry behind the result from Experiment 01.

---

## 2. Methodology

### Synthetic gradient representation

A base gradient \(B\) with shape \(64\times128\) is generated. A fixed bank of 256 random projection vectors is used for the LSH calculations.

The hashing implementation follows the relevant LSHFed conventions for flattening, projection, binary hashing, and Hamming-distance calculation.

### Case 1: Magnitude-only changes

The first test creates gradients by scaling the same base vector:

$$
g_s=sB.
$$

The direction of \(g_s\) is unchanged as \(s\) varies.

Euclidean distance changes because the vector magnitude changes. However, for positive \(s\), every projection keeps the same sign. The resulting hashes are therefore identical and their Hamming distance remains zero.

This isolates the part of Euclidean distance that LSH hashing does not capture.

### Case 2: Directional changes

The second test adds a random perturbation:

$$
g_\alpha=B+\alpha N,
$$

where \(N\) is a random perturbation and \(\alpha\) controls its strength.

As \(\alpha\) increases, the gradient changes direction. This can cause individual random projections to cross zero, producing different hash bits.

The Euclidean and Hamming distances are recorded as the perturbation strength increases.

### Case 3: Synthetic gradient population

The final test generates 200 gradients of the form

$$
g_i=B+\alpha_iN_i,
$$

using independently generated perturbations and perturbation strengths.

All

$$
\binom{200}{2}=19,900
$$

gradient pairs are compared.

The experiment first evaluates the complete 256-projection hashes and then repeats the correlation calculation for

$$
r\in\{1,2,4,5,8,16,32,64,128,256\}.
$$

The same projection bank is used throughout the sweep so that changing \(r\) does not introduce a new source of randomness.

---

## 3. Results and Interpretation

### Magnitude changes are not reflected in Hamming distance

Euclidean distance increases as the scale changes, while Hamming distance remains zero.

This shows that two gradients can be increasingly far apart in Euclidean distance while having exactly the same random-hyperplane hash, as long as their direction does not change.

### Directional changes are reflected in Hamming distance

When directional perturbations are introduced, Hamming distance starts changing along with Euclidean distance.

This shows that the LSH representation is responding to directional differences that cause gradients to cross projection hyperplanes.

Together, the first two tests explain an important part of the mismatch seen in Experiment 01: Euclidean distance includes magnitude differences that the sign-based hash does not directly represent.

### Synthetic population

The relationship is very different when the synthetic gradient population is generated through controlled directional perturbations.

At \(r=256\):

* Pearson correlation = 0.9944
* Spearman correlation = 0.9964

Correlation also improves steadily as \(r\) increases:

| \(r\) | Pearson | Spearman |
| ----: | ------: | -------: |
|     1 |  0.8830 |   0.8659 |
|     2 |  0.9295 |   0.9171 |
|     4 |  0.9660 |   0.9583 |
|     5 |  0.9722 |   0.9670 |
|     8 |  0.9818 |   0.9786 |
|    16 |  0.9872 |   0.9859 |
|    32 |  0.9910 |   0.9913 |
|    64 |  0.9927 |   0.9940 |
|   128 |  0.9938 |   0.9957 |
|   256 |  0.9944 |   0.9964 |

So, in this controlled synthetic setting, increasing \(r\) gives a progressively better approximation of the Euclidean ordering.

This also clarifies why the result from Experiment 01 cannot simply be described as an inherent failure of LSH to preserve distance. The relationship can be very strong when the gradient variation is dominated by directional changes. The within-model gradients from Experiment 01 have a different structure.

The next step is therefore to move from synthetic gradients to actual model updates produced by federated training. Experiment 03 tests the distance relationship in an LSHFed-style federated trajectory and, more importantly, checks whether the two distances lead to the same aggregator-selection decision.
