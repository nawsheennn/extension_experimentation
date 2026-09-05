# Direct Gradient Inversion Without Diffusion

## 1. Motivation / Objective

This experiment follows the diffusion-model and GSS diagnostics and shifts the investigation away from the diffusion pipeline entirely.

The earlier `unconditional_diffusion_manifold_diagnosis` showed that the FFHQ diffusion model referenced by the authors can successfully denoise Gaussian noise into natural face images. The subsequent `gss_with_no_guidance_diagnosis` showed that the active GSS sampling path also remains on the natural image manifold when the leaked-gradient guidance is removed. Together, these results make a broken diffusion prior or diffusion-to-guidance execution path less likely to explain the failed GGSS-R reconstruction.

This leaves a more fundamental question: 

> Does the leaked victim gradient itself contain enough information to recover the target image?

GGSS-R assumes that guiding a candidate toward the target gradient also constrains it toward the image that produced that gradient. The previous experiments showed that the gradient objective can be optimized, but the reconstructed image still remains far from the target. To test whether this problem exists independently of diffusion, this experiment removes the diffusion model and GGSS-R sampling machinery completely and performs direct gradient inversion in image space.

The experiment therefore tests whether a candidate initialized from random noise can recover the target simply by minimizing its victim-gradient difference from the leaked target gradient. If direct inversion succeeds, the gradient contains sufficient information for reconstruction and the remaining problem would lie elsewhere in GGSS-R. If the gradient can instead be matched closely while the image remains substantially different from the target, then gradient matching itself is not sufficient to identify the target under this victim setup.

## 2. Methodology / Pipeline / Implementation

The experiment reuses the exact single-epoch `MLP_1` victim checkpoint and class-0 CelebA training-member target established in `ggssr_unperturbed_reproduction_diagnostic`. Those inputs are treated as read-only and copied into a separate experiment directory. No diffusion repository, diffusion checkpoint, or GGSS-R implementation is loaded.

The victim is the authors' three-layer fully connected model:

$$
3\times256\times256
\rightarrow512
\rightarrow128
\rightarrow2,
$$

with 100,729,730 parameters. The target is resized to \(256\times256\), converted to a tensor, and normalized to \([-1,1]\), matching the preprocessing used in the previous reconstruction experiments. Its victim prediction is also verified as class 0 before the leaked target gradient is constructed.

Because `fc1.weight` alone contains more than 100 million gradient coordinates, matching the entire differentiable gradient at every optimization step would be unnecessarily expensive. The experiment therefore uses a fixed subset consisting of:

* 100,000 randomly selected `fc1.weight` coordinates;
* the complete `fc1.bias` gradient;
* the complete `fc2` gradients;
* the complete `fc3` gradients.

This gives **166,434 matched gradient coordinates** in total. The selected gradient is implemented analytically and verified against PyTorch autograd before reconstruction, ensuring that the optimized quantity is equivalent to the corresponding victim-model gradient.

For a candidate image \(x\), the inversion minimizes the normalized squared gradient difference:

$$
\mathcal{L}(x) = \frac{\text{mean}\left[(g(x)-g_{\text{target}})^2\right]}{\text{mean}(g_{\text{target}}^2)+10^{-12}}
$$

Rather than optimizing unrestricted pixel values directly, a free tensor \(z\) is optimized and converted to the candidate image using

$$
x=\tanh(z),
$$

which keeps every reconstructed pixel within the victim's expected \([-1,1]\) input range.

A single 1,500-step Adam run is first performed as a sanity check. The main experiment then repeats the inversion across **eight independent random restarts**, each using:

* 1,500 optimization steps;
* learning rate \(0.05\);
* the same fixed target gradient and selected coordinates;
* a different reproducible random initialization.

Gradient-space progress is measured using Euclidean gradient distance, relative gradient distance, and cosine similarity. Pixel MSE against the actual target is measured separately. This distinction is central to the experiment: the question is not only whether the leaked gradient can be matched, but whether matching it actually reconstructs the image that produced it.

## 3. Results / Outcomes / Observations / Interpretations

The direct gradient-inversion objective works as intended. In the initial sanity run, relative gradient distance decreases from **1.1363** to **0.5106**, while gradient cosine similarity increases from **0.4039** to **0.8598**. The optimizer is therefore clearly capable of moving the candidate toward the leaked target gradient without any diffusion machinery.

The corresponding improvement in image space is much smaller. Pixel MSE begins at **0.6683** and remains **0.6206** after 1,500 steps. The candidate becomes substantially more similar to the target in gradient space without becoming a successful reconstruction of the target image.

The eight independent restarts make this result considerably stronger.

The best gradient matches are runs 4 and 7. Their final relative gradient distances are only **0.0412** and **0.0433**, with cosine similarities of **0.99915** and **0.99906**. The reconstructed gradients are therefore almost perfectly aligned in direction with the leaked target gradient.

Despite this, their pixel MSEs remain **0.4311** and **0.4516**. None of the eight random restarts produces a low-error recovery of the target image. All eight final candidates are classified as class 0 by the victim, showing that the optimization can satisfy the target's class-related behavior without recovering its image identity.

The best gradient matches are also the best two reconstructions by pixel MSE, so gradient matching is not completely unrelated to image similarity. Moving closer to the target gradient can move the candidate in a useful image-space direction. The important result is that this relationship is not strong enough for reconstruction: even an almost exact gradient-direction match still corresponds to a substantially different image.

This rules out an important explanation for the earlier failure. The reconstruction problem is **not caused solely by the presence of the diffusion model or by GGSS-R's diffusion sampling procedure**. The same separation between gradient similarity and image recovery remains when diffusion is removed and the leaked gradient is attacked directly.

However, this experiment alone does not establish why that separation occurs or prove that the target gradient is numerically non-unique. It establishes the empirical problem: direct optimization can produce very strong gradient matches without recovering the target.

That result motivates `gradient_identifiability_analyses`, which examines the victim gradients across a larger image population, compares gradient-space similarity with image-space similarity, and analyzes the victim layer by layer to determine what information the leaked gradient is actually carrying.
