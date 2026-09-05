# GSS with Zero Gradient Guidance Diagnosis

## 1. Motivation / Objective

The previous `unconditional_diffusion_manifold_diagnosis.ipynb` showed that the authors' FFHQ diffusion model can independently denoise Gaussian noise into natural-looking human-face images.

However, that experiment does not test the actual GGSS-R reconstruction path. It bypasses the GSS conditioning mechanism completely.

The earlier full GGSS-R reproduction, on the other hand, uses gradient guidance and produced poor reconstruction results. This leaves an important implementation question:

> Is the reconstruction failure caused by the diffusion/GSS sampling path itself, or specifically by the gradient-guidance component?

This experiment isolates that question by running the authors' **GSS reconstruction pipeline with gradient guidance disabled**:

$$
m_r=0.
$$

Everything else is kept fixed as closely as possible:

- authors' repository and configuration,
- FFHQ diffusion checkpoint,
- victim-model checkpoint,
- CelebA target image,
- reconstruction measurement,
- GSS conditioning mechanism,
- 1000-step DDIM sampling.

The intended algorithmic change is only the removal of gradient guidance.

This creates a useful second control point.

If GSS with $m_r=0$ still produces meaningful face images, then the diffusion prior and the basic GSS sampling path are both capable of producing valid images. The poor result at $m_r>0$ would then become much less likely to be caused by a basic diffusion or GSS implementation failure.

The next investigation can consequently focus on the remaining explanation: whether the leaked victim gradient contains enough information to identify the target image.

---

## 2. Methodology / Pipeline / Implementation

### Isolated experiment environment

A completely separate workspace is created with dedicated directories for:

- the authors' repository,
- copied input artifacts,
- reconstruction outputs,
- experiment logs.

The inputs from the earlier unperturbed reproduction are treated as read-only sources. The FFHQ checkpoint, victim checkpoint, and target image are copied into the new workspace before execution.

This separation makes the ablation easier to reproduce and ensures that the earlier experiment is not modified.

### Authors' code and fixed inputs

A fresh checkout of the authors' repository is created, the `code.zip` folder is extracted and the required reconstruction files are checked, including:

- `sample_condition_same_inputs.py`
- `guided_diffusion/condition_methods.py`
- `guided_diffusion/measurements.py`
- `guided_diffusion/gaussian_diffusion.py`
- `guided_diffusion/attacked_model.py`
- the model and diffusion configuration files.

The FFHQ checkpoint is copied from the earlier reproduction and verified against its known SHA-256.

### Output-only compatibility patches

Two small patches are applied to the authors' runner.

The first creates the reconstruction progress directory before it is used.

The second reduces intermediate image saving so that selected states are stored every 100 diffusion steps rather than writing images at every step.

These changes affect only output handling. They do not change the GSS calculation or the diffusion update itself.

### Zero-guidance reconstruction

The authors' original reconstruction entry point is then executed with:

```text
--method GSS
--guidance_scale 0.0
```

The GSS machinery remains active, but the gradient-guidance coefficient is zero:

$$
m_r=0.
$$

Conceptually, the comparison is therefore:

$$
\text{Full GGSS-R: } m_r>0
$$

versus

$$
\text{GSS-only ablation: } m_r=0.
$$

The reconstruction still runs through the complete 1000-step diffusion trajectory. The experiment records the distance reported by the authors' code and saves sparse $x_t$ and $x_0$ states for visual inspection.

This is different from the unconditional diffusion experiment. The unconditional test directly calls the DDIM sampler without any GSS machinery, whereas this experiment deliberately keeps the GSS reconstruction path active and removes only its gradient-guidance contribution.

---

## 3. Results / Outcomes / Observations / Interpretations

The zero-guidance reconstruction completed successfully for all 1000 diffusion steps.

The run used:

```text
guidance_scale = 0.0
```

and took approximately 31.41 minutes. The final reported distance was approximately *60.8*.

The distance does not decrease monotonically. It fluctuates substantially throughout the trajectory and ends around 60.8. Therefore, the distance trace should not be interpreted as a conventional optimization curve where continuous decrease is expected. In this ablation, its main value is as a record of how the sampling process behaves when gradient guidance is disabled.

The more important result comes from the generated images.

The saved trajectory shows the diffusion process moving from noisy states toward coherent face structure. The intermediate $x_0$ estimates become increasingly interpretable, with clear face-like structure appearing through the trajectory. This demonstrates that the GSS-only path is capable of producing meaningful images even though the leaked gradient is no longer used to guide the diffusion process.

The run produced 22 PNG files, including the target/label image, the final reconstruction, and sparse $x_t$ and $x_0$ states at steps 900, 800, ..., 0.

As the diffusion process denoises the target image's noise, it produces a coherent image that is somewhat semantically similar to the target (female, face turned to the side, blonde, downturned mouth, shocked expression).
This is substantially better than the reconstruction experiments which used $m_r=0.2$ and reconstructed only patches of color with no discernible facial features.

It is understandable that the reconstructed image does not exactly match the target, as we deliberately did not provide gradient-guidance to the diffusion model.

This gives the experiment its main conclusion.

The GSS sampling path can run successfully with

$$
m_r=0
$$

and can produce natural, face-like outputs. Combined with the unconditional diffusion result, this makes two explanations less likely:

1. the FFHQ diffusion model is incapable of producing meaningful face images;
2. the basic GSS/diffusion sampling path fails simply because of the way it is implemented.

The remaining issue is therefore more specific.

When gradient guidance is enabled in the full GGSS-R reconstruction, the leaked victim gradient is supposed to steer the diffusion process toward the particular target image. The zero-guidance result shows that the sampler itself can reach the natural image manifold, but it does **not** show that the leaked gradient provides enough information to select the correct point on that manifold.

That is the key transition to the next diagnostic, `direct_gradient_inversion_wo_diffusion.ipynb`: remove diffusion altogether and ask whether the leaked gradient itself can reconstruct the target image. If direct gradient matching also produces candidates that are close in gradient space but far from the target in image space, the problem lies in the information carried by the leaked gradient.
