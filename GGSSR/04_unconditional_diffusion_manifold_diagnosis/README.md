# Unconditional Diffusion Manifold Diagnosis

## 1. Motivation / Objective

The GGSS-R reconstruction pipeline relies on the authors' FFHQ diffusion model as its image prior. When the full reconstruction produces poor images, one possible explanation is that the diffusion model itself is not generating meaningful images in the first place.

This experiment isolates that component completely.

The diffusion model is used **without** a victim model, target image, leaked gradient, measurement, GSS conditioning, or reconstruction objective. Starting only from Gaussian noise, the experiment asks a simple question:

> Can the authors' FFHQ diffusion model, using the original implementation and checkpoint, independently denoise random noise into natural-looking human-face images?

This is an important control experiment. If unconditional sampling works, then the poor reconstruction observed in the earlier GGSS-R experiments cannot simply be attributed to an incapable diffusion prior. The investigation can instead move to the interaction between the diffusion model and the leaked-gradient guidance.

The experiment also establishes a clean reference for the next diagnostic, `gss_with_no_guidance_diagnosis.ipynb`. There, the same diffusion machinery is placed back inside the authors' GSS reconstruction path, but the gradient-guidance coefficient is set to zero.

---

## 2. Methodology / Pipeline / Implementation

The experiment uses a fresh, isolated workspace so that the diffusion diagnostic does not modify the earlier reconstruction experiments.

### Authors' implementation and checkpoint

The authors' GGSS-R repository is cloned, the packaged `code.zip` is extracted and the required diffusion files are verified before the model is loaded.

The FFHQ diffusion checkpoint `ffhq_10m.pt` is copied into the experiment directory and verified using SHA-256.

The checkpoint is 357.07 MB and contains 362 parameter tensors. The loaded U-Net is also compared against the checkpoint parameter-by-parameter; all 362 tensors match with a maximum absolute difference of `0.0`.

### Diffusion configuration

The experiment keeps the authors' diffusion configuration rather than introducing a separate sampler implementation.

The main settings are:

- Image size: `256 × 256`
- Model: authors' FFHQ U-Net
- Sampler: DDIM
- Diffusion steps: `1000`
- Noise schedule: linear

The loaded U-Net contains 93,563,910 (93.5M) parameters.

### Unconditional sampling

Four independent trajectories are generated.

For each sample, the initial state is pure Gaussian noise:

$$
x_{999} \sim \mathcal{N}(0,I).
$$

The experiment then traverses the authors' DDIM trajectory backwards:

$$
x_{999}\rightarrow x_{998}\rightarrow\cdots\rightarrow x_0.
$$

At every step, the authors' own sampler.p_sample() implementation is called.

No conditioning function is invoked:

- no victim model,
- no target image,
- no gradient,
- no measurement,
- no GSS,
- no gradient guidance.

Selected `x_t` states and `pred_xstart` estimates are saved every 100 steps so that the complete denoising trajectory can be inspected without storing every intermediate image.

For visualization, the primary conversion is the fixed model-range mapping

$$
x_{\mathrm{image}}=\frac{x+1}{2}.
$$

This converts the diffusion-generated images with pixel range $[-1, 1]$ to a displayable $[0, 1]$ format. The authors' `clear_color()` min-max normalization is not used as the primary representation because independently rescaling each image can make poor absolute scaling appear visually better than it actually is.

Finally, the four generated samples are collected into a contact sheet and an experiment manifest records the implementation, checkpoint, configuration, seed, device, and runtime.

---

## 3. Results / Outcomes / Observations / Interpretations

All four unconditional diffusion trajectories completed successfully. Each required the full 1000-step DDIM trajectory and took approximately 1.7 minutes on the Tesla T4 used for the experiment.

The generated samples are natural-looking human-face images rather than unresolved noise. The four outputs also show that the behavior is not limited to a single successful random seed: all four independent trajectories reach coherent face images.

The trajectory inspection provides the same conclusion from another perspective. Starting from Gaussian noise, the diffusion process progressively removes noise and produces structured human-face content. The final contact sheet shows all four independent samples together.

Therefore, this diagnostic supports the diffusion prior as a working component of the pipeline.

This does not prove that the diffusion prior will reconstruct the correct target when conditioned on a leaked gradient. It only establishes that, when used by itself, the model can successfully move random noise onto a meaningful human-face image manifold.

That distinction is important for the next step. The full GGSS-R failure cannot be explained simply by saying that the diffusion model is incapable of producing faces. The next diagnostic therefore keeps the same diffusion model but places it back inside the authors' GSS reconstruction pipeline while setting the gradient-guidance coefficient to

$$
m_r=0.
$$

If that GSS-only path also produces meaningful face images, the remaining question becomes much more focused. That is, is the problem caused by the leaked gradient not providing enough image-specific information for successful reconstruction?
