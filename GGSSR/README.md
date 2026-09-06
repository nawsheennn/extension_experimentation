# GGSS-R Reconstruction Diagnostics and Extensions

## Overview

This directory contains a sequence of experiments investigating GGSS-R, the gradient inversion attack proposed in *Enhanced Privacy Leakage from Noise-Perturbed Gradients via Gradient-Guided
Conditional Diffusion Models*. GGSS-R reconstructs a client's private training image from its leaked model gradient by combining two signals during reverse diffusion:

* a **diffusion prior**, which keeps the generated sample on the natural image manifold;
* **gradient guidance**, which steers the sample toward an image whose victim-model gradient matches the leaked gradient.

The paper evaluates reconstruction across several victim architectures and introduces Reconstruction Vulnerability (RV) to quantify how vulnerable different models are to gradient inversion. It also studies reconstruction from perturbed gradients and uses a fixed guidance rate $m_r$ to balance gradient matching against the diffusion prior.

The experiments in this directory examine two aspects of that setup more closely.

### Victim architecture is only part of reconstruction vulnerability.

The paper reports reconstruction vulnerability for different model architectures, but the released repository does not include the corresponding victim checkpoints or fully specify the trained state associated with those reconstruction results. Recreating the victim locally exposed an important additional factor: **the same architecture can be either highly reconstructable or effectively non-reconstructable depending on its training state and resulting gradient structure.**

A gradient can be nonzero and its distance can be successfully minimized without containing enough image-specific information to recover the input. Across the diagnostic experiments here, reconstruction failure is traced through the pipeline until the victim gradient itself is isolated. Layer-wise analyses then show that what matters is not simply model size, classification loss, or total gradient magnitude, but whether a sufficiently strong and informative gradient reaches the input-facing layers.

This extends the architecture-focused view of reconstruction vulnerability into an architecture-and-state-dependent view of gradient identifiability.

### Gradient similarity is not necessarily image similarity.

GGSS-R guides reconstruction by reducing the distance between candidate and leaked gradients. Several experiments here test the assumption underlying that objective: whether moving closer in gradient space actually constrains the candidate toward the private image.

For poorly identifying victim states, it does not. Candidates can reach extremely high gradient cosine similarity while remaining far from the target in pixel and perceptual space. Population-level and layer-wise analyses are used to examine why this happens and which victim states instead produce gradients from which the image can genuinely be recovered.

### Static guidance can overfit a perturbed gradient.

The paper also shows that under sufficiently noisy gradients, reconstruction quality can improve and then deteriorate during reverse diffusion. The authors attribute this behavior to gradient guidance eventually overpowering the diffusion prior and forcing the reconstruction toward the noise contained in the perturbed gradient.

The released method nevertheless uses a **static guidance rate** throughout reconstruction.

The final experiment extends this mechanism with a **linearly decaying guidance schedule**. Strong guidance is retained early to establish target correspondence, then reduced during later denoising so that the natural-image prior is less dominated by the noisy gradient. Under the tested Gaussian perturbation, this produces substantially better reconstruction than the paper's fixed guidance rate.

---

## Experimental Progression

The experiments are intended to be read as a sequence. Each one isolates a question raised by the result immediately before it.

### Baseline GGSS-R reconstruction

The sequence begins with a direct reconstruction attempt using the authors' released GGSS-R pipeline. Their `MLP_1` victim is trained locally for one epoch, while a CelebA test image is used as the reconstruction target with the paper's fixed \(m_r=0.2\). The pipeline executes successfully, but the target is not recovered visually. This establishes the initial failure and motivates testing whether the victim and target setup is responsible.

### More training and a training-member target

The next experiment trains the same victim for three epochs and replaces the held-out target with an explicitly class-0 image from the victim's training set. The expectation is that repeated exposure to the target may make its gradient more image-specific. Instead, reconstruction becomes substantially worse. More victim training and guaranteed training-set membership therefore do not solve the problem and suggest that the victim's learned state may itself be affecting reconstruction.

### Unperturbed reproduction and pipeline diagnostics

The reconstruction is then returned to the single-epoch victim while replacing the generic diffusion checkpoint used earlier with the exact checkpoint referenced by the authors. The attack is kept unperturbed and instrumented with additional diagnostics. Reconstruction still fails, shifting the investigation away from simple target selection and checkpoint mismatch and toward isolating the individual components of GGSS-R.

### Unconditional diffusion diagnosis

The gradient-guidance path is removed entirely and the referenced FFHQ diffusion model is tested on its own. Starting from Gaussian noise, it successfully produces natural face images. The diffusion prior is therefore functional and capable of representing the image domain required by the attack.

### GSS without leaked-gradient guidance

The active GSS sampling machinery is then retained while the leaked-gradient guidance is disabled. The resulting samples remain on the natural face manifold even though, as expected, they do not reconstruct the private target. Together with the unconditional diffusion test, this makes a broken diffusion model or basic GSS sampling path unlikely to explain the reconstruction failures.

### Direct gradient inversion without diffusion

Attention then shifts entirely to the leaked gradient. Diffusion is removed and the candidate image is optimized directly against the victim gradient. Gradient distance decreases substantially and cosine similarity can exceed 0.999, yet the reconstructed image remains far from the target. This establishes that the earlier failures are not solely caused by diffusion: under this victim state, matching the gradient itself is insufficient for image recovery.

### Gradient identifiability analysis

The gradient is then studied directly across a balanced population of CelebA images and across individual victim layers. Gradient similarity is found to contain real image-related structure, but not enough to uniquely constrain the target under the tested single-epoch `MLP_1`. Most importantly, the input-facing layer carries very weak target-specific gradient signal compared with downstream classification layers. This provides a concrete explanation for why gradient optimization can succeed numerically while image reconstruction fails.

### Victim architecture and training-state comparison

The victim is next treated as the experimental variable. Four custom architectures are compared at initialization and after one epoch of training using direct inversion of clean gradients. The untrained 25.2M-parameter MLP reconstructs the target consistently across all seeds, reaching a mean MSE of **0.00113** and PSNR of **29.72 dB**. After one epoch, the same architecture becomes much harder to reconstruct. All three CNN architectures fail even at initialization: their total gradients may be large, but the gradient signal largely vanishes before reaching their early feature layers. This demonstrates that reconstruction vulnerability depends on **both architecture and model state**, specifically on whether image-identifying gradients survive to input-connected layers.

### Scheduled guidance for perturbed gradients

With a victim state whose clean gradient has been shown to support reconstruction, the final experiment returns to GGSS-R's intended perturbed-gradient setting. At Gaussian noise variance \(0.01\), the paper's fixed \(m_r=0.2\) is compared with a linear schedule that decays from 0.2 to 0 throughout reverse diffusion. The scheduled run increasingly outperforms static guidance as reconstruction progresses, finishing at **0.0548 MSE, 12.61 dB PSNR, and 0.8744 LPIPS**, compared with **0.1087 MSE, 9.64 dB PSNR, and 0.9317 LPIPS** for fixed guidance. The result shows that reducing gradient influence during later denoising can better preserve reconstruction quality when the leaked gradient itself is noisy.

---

## Overall Summary

The experiments separate two questions that can otherwise become conflated in gradient-guided reconstruction:

1. Does the victim gradient contain enough image-specific information to reconstruct the input?
2. If it does, how should that gradient be used when it has been perturbed?

The diagnostic sequence shows that the first question depends strongly on the victim's architecture, training state, and layer-wise gradient propagation. Successful optimization of a gradient-matching objective is not by itself evidence that the private image is recoverable.

Once a demonstrably identifiable victim gradient is established, the perturbed-gradient experiment addresses the second question. There, replacing GGSS-R's fixed guidance strength with a decreasing schedule improves the balance between target guidance and the diffusion prior.

Together, these experiments extend the GGSS-R evaluation in two directions: characterizing reconstruction vulnerability through gradient identifiability rather than architecture alone, and introducing scheduled guidance as an alternative to static guidance for reconstruction from perturbed gradients.
