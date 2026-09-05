# GGSS-R Unperturbed Reconstruction Diagnostic

## 1. Motivation / Objective

This experiment is the first major diagnostic step after the two initial reconstruction attempts.

The baseline experiment used a one-epoch victim and a held-out target, while the second experiment used a three-epoch victim and a training-member target. Neither produced the expected reconstruction. The second experiment in particular showed that simply training the victim longer and giving the attack a target that it had actually seen during training did not solve the problem.

At this point, changing the victim-training conditions further would not tell us whether the problem lies elsewhere in the reconstruction pipeline. We therefore return to a controlled GGSS-R reproduction and inspect the individual components before and after the actual reconstruction.

A further issue in the baseline setup also needs to be addressed. The authors' GGSS-R repository does not itself contain the required diffusion checkpoint. This experiment therefore uses the **FFHQ diffusion checkpoint referenced by the paper**, rather than treating an arbitrary public diffusion checkpoint as interchangeable with the one expected by the authors' pipeline.

The main objective is to answer a more basic question:

> Is the GGSS-R reconstruction pipeline actually receiving valid gradients, computing the gradient-distance objective correctly, and propagating that objective back to the diffusion image as intended?

The experiment does not assume that the final reconstruction must succeed. Instead, it establishes whether the relevant pieces of the pipeline are functioning before interpreting the eventual reconstruction failure.

This is important for the later diagnostic sequence. If the diffusion model itself is not generating valid images, that needs to be isolated. If the victim gradient is invalid, that needs to be isolated. If the gradient-distance objective cannot be differentiated with respect to the image, the GGSS-R guidance mechanism cannot work. And if all of these checks pass but reconstruction still fails, attention can move toward the information contained in the leaked gradient itself.

## 2. Methodology / Pipeline / Implementation

The experiment is built around the authors' released GGSS-R implementation rather than reimplementing the attack from scratch.

The public repository is cloned and its `code.zip` bundle is extracted. The notebook records the upstream commit and verifies that the expected runner, diffusion implementation, victim model, measurement operator, and configuration files are present.

The FFHQ diffusion checkpoint `ffhq_10m.pt` is obtained from the public checkpoint location referenced by the DPS project. The checkpoint is loaded and verified before being placed at the path expected by GGSS-R.

The victim follows the authors' `MLP_1` architecture and is trained using the established experiment procedure. A deterministic class-0 CelebA training-member target is constructed and stored as an immutable target artifact. The target and victim checkpoint are then verified together before reconstruction.

Three small compatibility/output patches are applied to the authors' source:

1. create the reconstruction progress directory before it is written to;
2. create the predicted-\(x_0\) directory before those snapshots are written;
3. save sparse trajectory snapshots rather than writing an image at every one of the 1000 diffusion steps.

These patches change file handling only and are not intended to modify the GGSS-R reconstruction equations.

Before running the attack, the notebook performs five diagnostics.

### Target-gradient validity

The target image is passed through the victim model and its full parameter gradient is computed. The diagnostic checks the gradient norm, the number of nonzero elements, and whether the autograd graph is preserved.

This establishes that the target produces a valid measurement for GGSS-R.

### Unrelated-image gradient validity

A different CelebA image is passed through the same victim and loss function. Its gradient is compared at the basic numerical level with the target gradient.

The purpose is to make sure the target is not an unusual case where the victim-gradient mechanism itself is malfunctioning. A normal unrelated image should also produce a valid, nonzero gradient.

### Second-order differentiability

GGSS-R does not only require the victim gradient

$$
g(x)=\nabla_\theta L(f_\theta(x),y).
$$

It needs to differentiate a distance between the target gradient and the candidate gradient back through the candidate image. Conceptually, the required path is

$$
x
\rightarrow
g(x)
\rightarrow
\|g(x)-g_{\mathrm{target}}\|
\rightarrow
\nabla_x
\|g(x)-g_{\mathrm{target}}\|.
$$

The notebook explicitly constructs this path and checks that the resulting image gradient is finite and nonzero.

### Gradient behavior on an actual diffusion sample

The previous checks use real CelebA images. GGSS-R, however, applies the gradient guidance to images produced by the FFHQ diffusion process.

The notebook therefore generates an unconditional 1000-step DDIM sample using the same FFHQ diffusion model and checks whether this generated image also produces a valid victim gradient and a differentiable gradient-distance objective.

This isolates the interaction between the diffusion image representation and the victim-gradient measurement before the full guided reconstruction is attempted.

### Original GGSS-R measurement operator

Finally, the same diffusion sample is passed through the authors' original `ReconstructionOperator.forward()` implementation.

Its gradient output and target-to-sample gradient distance are compared against the values obtained from the controlled diagnostic implementation. Exact agreement is used as a consistency check that the authors' measurement operator is computing the expected quantity.

After these checks, the notebook executes the original GGSS-R reconstruction using GSS conditioning, 1000 DDIM steps, and

$$
m_r=0.20,
$$

with no additional gradient perturbation.

The reconstruction trajectory is then analyzed using MSE, PSNR, LPIPS, victim-gradient distance, relative gradient distance, and gradient cosine similarity. The post-reconstruction analysis also decomposes the remaining gradient mismatch by victim-model layer.

## 3. Results / Outcomes / Observations / Interpretations

The pre-reconstruction diagnostics consistently passed.

The target produced a nonzero full parameter gradient with norm approximately 1.3358.

There were 17,107,415 nonzero gradient elements out of 100,729,730, and the autograd graph was preserved.

The unrelated CelebA image also produced a valid gradient, with norm approximately 35.28 and more than 22.6 million nonzero elements. This shows that the victim-gradient measurement is not failing only for the target.

The second-order differentiability test also passed. The gradient-distance objective had value approximately 35.86, and its gradient with respect to the image had norm approximately 11.68, with all 196,608 input pixels receiving a nonzero gradient. This is important because it demonstrates that the gradient-matching objective can actually propagate information back to the candidate image.

The diffusion-specific diagnostic also passed. The DPS FFHQ model successfully generated a 256×256 sample through the full 1000-step DDIM process. That generated image produced a valid victim gradient with norm approximately 2.839, and its gradient differed from the target gradient by approximately 2.801. The gradient-distance objective also propagated a nonzero gradient back to the diffusion image.

Most importantly, the authors' original `ReconstructionOperator` reproduced the same gradient-distance result: 2.8006341457 with an absolute difference of 0 from the controlled diagnostic calculation.

These checks substantially narrow down the possible sources of failure. The target gradient is valid, unrelated images produce valid gradients, the gradient-distance objective is differentiable with respect to image pixels, the diffusion model produces usable samples, and the authors' original measurement operator agrees with the controlled implementation.

The full unperturbed GGSS-R reconstruction nevertheless fails to recover the target.

The 1000-step run completes, but the reconstruction trajectory does not approach the target in image space. The final reconstruction has:

* pixel MSE: **0.173091**
* PSNR: **7.617 dB**
* LPIPS: **0.688771**
* victim-gradient distance: **0.428106**
* relative gradient distance: **0.320480**
* gradient cosine similarity: **0.948341**

The reconstruction therefore does achieve a reasonably strong alignment with the target gradient, but this does not translate into a visually accurate reconstruction. The saved trajectory shows the same separation: by the final step, the gradient cosine similarity is high while the image remains substantially different from the target.

The comparison with three unrelated class-0 CelebA images is also informative. Those images have pixel MSEs around **0.23–0.24** from the target, while their relative gradient distances are much larger, approximately **23–87**. The GGSS-R reconstruction is considerably closer to the target in gradient space than these comparison images, but it is still far from the target in image space.

The layer-wise decomposition shows that the remaining target-to-reconstruction gradient mismatch is concentrated mainly in:

* `fc2.weight`: **74.99%**
* `fc1.weight`: **21.86%**
* `fc3.weight`: **3.15%**

with the bias terms contributing negligibly.

Taken together, these results change the direction of the investigation. The reconstruction failure is not readily explained by an invalid target gradient, a broken second-order gradient path, an unusable diffusion sample, or a mismatch between the controlled measurement and the authors' `ReconstructionOperator`.

The remaining question is therefore not simply whether GGSS-R is capable of minimizing gradient distance. It clearly can. The more important question is whether minimizing the leaked-gradient distance actually provides enough information to identify the target image.

This motivates the next diagnostics. The diffusion model and the GSS execution path are first tested independently in `unconditional_diffusion_manifold_diagnosis` and `gss_with_no_guidance_diagnosis`. Direct gradient inversion is then tested without diffusion in `direct_gradient_inversion_wo_diffusion`. Finally, `gradient_identifiability_analyses` examines the relationship between image space and gradient space more systematically to determine whether the leaked gradient from the MLP victim is sufficiently image-specific for reconstruction.
