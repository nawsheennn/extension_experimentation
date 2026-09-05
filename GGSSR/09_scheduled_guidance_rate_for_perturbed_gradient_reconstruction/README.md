# Scheduled Guidance Rate for Perturbed Gradient Reconstruction

## 1. Motivation / Objective

GGSS-R reconstructs private images by coupling gradient guidance with reverse diffusion. The diffusion model acts as a natural-image prior, while the leaked gradient guides the generated image toward the private target.

The paper uses a fixed guidance rate \(m_r\) to balance these two components and reports **\(m_r=0.2\)** as the best static setting. Its noisy-gradient experiments, however, show an important limitation of keeping that balance unchanged throughout reconstruction. With moderate Gaussian perturbation, reconstruction quality can reach a strong intermediate state and then deteriorate during later diffusion steps. Because the leaked gradient itself contains noise, continuing to enforce it strongly can make the reconstruction increasingly follow the noise-contaminated gradient rather than the clean target.

Early stopping does not provide a practical solution: an attacker does not have the private target image and therefore cannot know when the best reconstruction has been reached.

This experiment extends GGSS-R by making the guidance rate **timestep-dependent**. The reconstruction begins with strong gradient guidance to steer the sample toward the target, then progressively reduces that influence so that the diffusion prior has more control during later refinement.

The paper's fixed setting,

$$
m_r=0.2,
$$

is compared with a linear schedule,

$$
m_r(i)=0.2\frac{i}{999},
$$

for reverse-diffusion indices $i=999,\ldots,0$. The scheduled run therefore starts at $m_r=0.2$ and ends at $m_r=0$.

The experiment specifically tests reconstruction from a gradient perturbed with **Gaussian noise variance \(0.01\)**. The question is whether decaying $m_r$ can preserve useful early gradient guidance while reducing the late-stage influence of the perturbation.

## 2. Methodology / Pipeline / Implementation

The experiment retains the authors' GGSS-R reconstruction pipeline and their referenced FFHQ diffusion checkpoint.

The victim is the paper's **MLP-3 architecture**, called `MLP_1` in the authors' implementation, with approximately 100.7M parameters. It is used at **epoch 0**, directly after initialization, so that the experiment starts from a victim whose clean gradient is suitable for reconstruction rather than introducing weak gradients caused by training.

A CelebA target is prepared at the $256\times256$ resolution required by the victim and reconstruction pipeline. Its clean victim gradient \(g\) is calculated and perturbed before reconstruction:

$$
\tilde g=g+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,0.01).
$$

Both experimental conditions receive exactly the same perturbed gradient.

Two 1,000-step GGSS-R reconstructions are then run. The **static condition** uses $m_r=0.2$ at every reverse-diffusion step. The **scheduled condition** begins at the same value and linearly decreases $m_r$ to zero over the trajectory.

The scheduling is introduced through a small hook in the authors' GSS implementation rather than by changing the underlying gradient-conditioning procedure. Both runs use the same target, victim initialization, diffusion checkpoint, noisy gradient, and initial diffusion noise \(x_T\). This keeps the guidance-rate policy as the experimental difference.

Reconstruction quality is measured directly against the target using:

* **MSE**, where lower is better;
* **PSNR**, where higher is better;
* **LPIPS**, where lower is better.

Metrics are recorded throughout the full trajectory rather than only at the final output. Sparse reconstruction snapshots, metric curves, comparison figures, and a 10×2 contact sheet are also generated to compare how the two reconstructions evolve.

Image-space metrics are used as the measure of reconstruction quality because the attack is deliberately matching a perturbed gradient. A closer match to that noisy gradient is not necessarily a closer reconstruction of the clean target.

## 3. Results / Outcomes / Observations / Interpretations

The scheduled guidance run produces the better reconstruction, with the advantage becoming progressively clearer during the second half of reverse diffusion.

By diffusion index 500, the scheduled condition is already better across all three reconstruction metrics:

| Guidance                     |      MSE ↓ |      PSNR ↑ |    LPIPS ↓ |
| ---------------------------- | ---------: | ----------: | ---------: |
| Static \(m_r\), index 500    |     0.1553 |     8.09 dB |     0.9369 |
| Scheduled \(m_r\), index 500 | **0.1143** | **9.42 dB** | **0.9218** |

The separation is considerably larger at the end of the 1,000-step trajectory:

| Guidance                 |      MSE ↓ |       PSNR ↑ |    LPIPS ↓ |
| ------------------------ | ---------: | -----------: | ---------: |
| Static \(m_r\), final    |     0.1087 |      9.64 dB |     0.9317 |
| Scheduled \(m_r\), final | **0.0548** | **12.61 dB** | **0.8744** |

Scheduled guidance therefore ends with approximately **half the MSE** of fixed guidance and nearly **3 dB higher PSNR**, alongside a lower LPIPS score.

The full metric trajectories show that the final result is part of a consistent trend rather than an isolated improvement. The two runs remain comparatively close during early denoising, when their guidance rates are still similar. As the scheduled \(m_r\) decreases, a clear gap develops between the curves. Scheduled guidance achieves lower MSE, higher PSNR, and lower LPIPS through the later reconstruction stages.

The image trajectories support the numerical results. Both runs progressively recover facial structure, but the fixed-guidance samples become visibly more susceptible to the noisy gradient. In the 10×2 contact sheet, the scheduled reconstruction remains cleaner as reverse diffusion proceeds. The difference is already visible around the halfway point and is stronger in the final reconstruction.

The scheduled condition also achieves better best-observed MSE and PSNR.

The results indicate that a guidance strength useful at the beginning of noisy-gradient reconstruction can become too influential during later refinement. With fixed $m_r=0.2$, the perturbed leaked gradient continues to exert the same guidance strength throughout the trajectory. Linear decay gradually transfers more control back to the diffusion prior, reducing the influence of the noise-contaminated gradient as reconstruction progresses.

For the tested **Gaussian noise variance \(0.01\)**, scheduling $m_r$ from 0.2 to 0 gives a clear improvement over the paper's fixed $m_r=0.2$. The experiment therefore identifies **guidance scheduling as a useful extension to GGSS-R under gradient perturbation**.
