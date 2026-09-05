# Gradient Identifiability Analyses

## 1. Motivation / Objective

This experiment is the final diagnostic in the reconstruction sequence. It investigates **why minimizing the leaked-gradient difference has repeatedly failed to recover the target image**.

The preceding experiments progressively isolated the main components of GGSS-R. 
`unconditional_diffusion_manifold_diagnosis` established that the diffusion model can generate natural face images from noise. 
`gss_with_no_guidance_diagnosis` showed that the active GSS sampling path also remains capable of producing images on the natural manifold without gradient guidance, even though the reconstructed image is **not** similar to the target due to the absence of gradient guidance. 
`direct_gradient_inversion_wo_diffusion` then removed diffusion entirely and optimized an image directly against the leaked victim gradient.

That direct inversion produced the key motivation for this experiment. Some reconstructed candidates achieved relative gradient distances near **0.04** and gradient cosine similarities above **0.999**, yet still remained far from the target in pixel space. The problem therefore appears deeper than diffusion: **a candidate can become extremely close to the leaked gradient without recovering the image that produced it.**

This experiment examines whether the single-epoch `MLP_1` victim gradient is sufficiently image-specific for reconstruction in the first place.

A useful leaked gradient must do more than vary across inputs or encode the victim's binary `Smiling` classification. For image reconstruction, gradient similarity must meaningfully constrain image similarity, and the gradient must retain enough target-specific information to distinguish the original image from alternative inputs.

The experiment therefore studies gradient identifiability at three levels:

* how much individual gradient coordinates vary across different images;
* how gradient-space relationships correspond to class and image-space relationships across a population;
* where target-specific gradient information is distributed across the victim model's layers.

The goal is not to assume in advance that the gradient is non-identifying. It is to determine what the measured gradient structure can actually tell us about the reconstruction failures observed throughout the preceding experiments.

## 2. Methodology / Pipeline / Implementation

The experiment uses the same single-epoch `MLP_1` victim checkpoint from `ggssr_unperturbed_reproduction_diagnostic`. The original GGSS-R target and saved reconstruction trajectory are also reused where target-specific comparisons are required.

The victim contains **100,729,730 parameters**, making it impractical to retain complete gradients for hundreds of images simultaneously. 
The analysis therefore combines memory-efficient population representations with exact full-gradient calculations where they are most important.

A balanced population of **200 CelebA images** is sampled using a fixed seed:

* 100 images with `Smiling == 0`;
* 100 images with `Smiling == 1`.

Every image is passed through the same victim and differentiated using the same classification objective.

For per-coordinate analysis, **100,000 fixed gradient coordinates** are sampled and observed across all 200 images. 
Their variance and sign entropy are measured to determine how strongly individual gradient elements change across inputs.

For population-wide pairwise analysis, each full gradient is compressed into the same **256-dimensional random projection**. 
Each projected dimension is constructed from 4,096 gradient coordinates, so the projection samples information from approximately one million coordinates overall. 
This compact representation makes it possible to perform pairwise comparisons, PCA, clustering, and nearest-neighbour analysis without storing the complete 100-million-dimensional gradient population.

Two gradient-space measures are used:

* Euclidean distance, which measures absolute separation;
* cosine similarity, which measures directional agreement;

The 200 projected gradients are then analyzed in several complementary ways.

First, same-class and different-class image pairs are compared to determine whether the gradient contains class-related structure. 
PCA and K-means clustering are then used to characterize broader population structure and redundancy. 
Gradient-space nearest neighbours test whether images with highly similar gradients necessarily share the same class or image characteristics.

The experiment next computes pixel MSE for all **19,900 (200C2) unique image pairs** and directly compares image-space distance with gradient-space similarity. 
Spearman correlations quantify whether images that are closer in gradient space also tend to be closer in image space.

The original GGSS-R target is then reintroduced for a more exact analysis. 
Unlike the population-wide projected analysis, its gradient is compared with all 200 sampled alternatives using the **full victim gradient**. 
This checks whether images nearest to the target in exact gradient space are also visually close to it.

Finally, the target comparison is decomposed across the victim's six parameter tensors:

```text
fc1.weight
fc1.bias
fc2.weight
fc2.bias
fc3.weight
fc3.bias
```

This is especially important for `MLP_1`. Its `fc1.weight` tensor contains **100,663,808 parameters**, or about **99.9% of the entire model**, and directly receives the flattened image as input. 
The layer-wise analysis therefore tests whether this extremely large input-facing layer actually carries a strong target-dependent gradient signal or whether the leaked gradient is dominated by the much smaller downstream classification layers.

The saved GGSS-R reconstruction trajectory is also evaluated against the target in both image and exact gradient space. 
This connects the population-level diagnosis back to the reconstruction failure that motivated the experiment.

## 3. Results / Outcomes / Observations / Interpretations

The results show that the victim gradient is **not devoid of image information**, but the information it contains is **not sufficiently tied to image identity** to make gradient matching equivalent to image recovery.

At the individual-coordinate level, gradient variation is limited across much of the sampled population. 
Among the 100,000 inspected coordinates, the median variance is only **\(6.28\times10^{-8}\)** and the median sign entropy is **0.242 bits**. Only **10.1%** of coordinates have sign entropy above 0.9 bits. 
Most sampled coordinates therefore change relatively little or retain a stable sign across different images, while stronger variation is concentrated in a much smaller subset.

The projected population nevertheless contains real structure. Same-class image pairs have a slightly higher mean gradient cosine similarity than different-class pairs (**0.5165 vs. 0.4819**) and a lower mean projected-gradient distance (**0.00143 vs. 0.00196**). 
The gradients therefore contain some information related to the victim's `Smiling` classification, but the same-class and different-class distributions overlap substantially.

The PCA results show that this projected gradient population is also highly redundant: **16 components explain 95%** of the variation in the 256-dimensional representation, and 39 explain 99%. 
K-means finds its best tested silhouette score at two clusters (**0.4035**), indicating broad population structure but not cleanly separated groups. 
Because the clustering is unsupervised and is not matched directly to the labels, this does not establish that the two clusters correspond to the two `Smiling` classes.

The nearest-neighbour analysis further shows that high gradient similarity is not simply class identity. 
For one class-0 query, the closest gradient-space neighbour is a **class-1** image with cosine similarity 0.9713. 
Gradient space therefore captures input-dependent structure beyond the binary label, but a highly similar gradient does not uniquely determine even the image's class, much less its exact visual identity.

The population-wide image comparison gives a more nuanced result. Gradient cosine similarity has a substantial negative Spearman correlation with pixel MSE:

$$
\rho=-0.6348.
$$

Images with more directionally similar gradients therefore tend, in general, to be more similar in pixel space. Gradient information is not arbitrary or disconnected from image content.

Raw gradient Euclidean distance is much less informative, with

$$
\rho=0.1250
$$

against pixel MSE. Directional gradient agreement is therefore much more closely associated with image similarity than absolute projected-gradient distance in this population.

Crucially, however, this population-level relationship is not a one-to-one mapping. 
The exact target comparison shows that images closest to the target in full-gradient Euclidean distance or even cosine similarity can still have large pixel errors. 
The closest of the 200 sampled alternatives has full-gradient distance **1.1422**, but its pixel MSE from the target is still **0.6008**. 
The sampled population does not contain another image with an effectively identical full gradient, so it is not that gradients are completely non-unique. 
What they establish is that gradient closeness is not a reliable proxy for visual identity within the tested population under the current victim model setup.

The layer-wise analysis provides the strongest explanation for why this problem occurs under the current victim model.

Although `fc1.weight` contains approximately **99.9% of all victim parameters** and is the layer most directly connected to the input pixels, its gradient shows very weak target-relative directional structure across the population.
Its mean cosine similarity with the target gradient is only **0.0503**. In contrast, the much smaller `fc3.weight` has mean target cosine **0.6024**, while `fc2.weight` and `fc3.weight` dominate the raw target-gradient differences.

This means that the leaked gradient signal is distributed very unevenly. 
The enormous input-facing layer that has the greatest capacity to retain image-specific information contributes relatively little useful gradient signal, while the smaller downstream layers associated more directly with the classification decision contribute much more strongly to the gradient objective.

The victim's behavior is consistent with this finding. The single-epoch model already classifies the target confidently and produces a very small target loss. The measured layer-wise gradients indicate that the backpropagated signal has become extremely weak by the time it reaches `fc1`. 
This provides a plausible mechanism for the observed identifiability problem: the model can be a successful `Smiling` classifier while its leaked gradient carries relatively little signal through the layer that is most directly tied to individual pixels.

The saved GGSS-R trajectory confirms the practical consequence. 
As reconstruction progresses, gradient and image metrics do improve together: gradient distance falls from **0.8918** to **0.2898**, cosine similarity rises to **0.9764**, and pixel MSE falls from **0.8997** to **0.6085**. 
Gradient guidance is therefore moving the reconstruction in a direction that is somewhat informative about the target.

But the improvement saturates far from actual recovery. Even with cosine similarity of **0.9764**, the final saved reconstruction still has MSE **0.6085** and LPIPS **0.7490**. 
This is the same pattern observed more strongly in direct gradient inversion, where cosine similarity could exceed 0.999 without producing a recognizable target reconstruction.

Taken together, the diagnostic sequence points to a specific limitation of this reproduction setup. 
The diffusion model can generate natural images, the GSS execution path can operate on that manifold, the gradient objective can propagate through the candidate image, and direct gradient optimization can achieve extremely strong gradient alignment. 
The remaining failure is that **strong gradient alignment under the single-epoch `MLP_1` victim does not constrain the candidate tightly enough to recover the original image**.

The results therefore shift the next experimental question from the GGSS-R sampler to the victim model itself. 
A useful follow-up is to test whether a victim architecture or training state that preserves a stronger gradient signal in its input-facing layers produces a tighter relationship between gradient similarity and image identity, and consequently enables better reconstruction.
