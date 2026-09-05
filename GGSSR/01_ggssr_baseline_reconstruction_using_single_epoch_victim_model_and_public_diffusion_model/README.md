# GGSS-R Baseline: Single-Epoch Victim and Public Diffusion

## 1. Motivation / Objective

This is the first reconstruction experiment in the diagnostic sequence. Its purpose is to establish a baseline reproduction of the GGSS-R image reconstruction pipeline using the authors' released implementation.

The authors' method reconstructs an image from a victim model's leaked gradient by combining a diffusion prior with gradient-guided sampling. Before investigating why the method may fail, we first need to reproduce the complete pipeline as closely as possible and determine whether the reconstruction works under a reasonable implementation of the authors' setup.

There are two important limitations in the released materials. The authors' repository does not provide the trained victim-model checkpoint used in their experiments, and the required FFHQ diffusion checkpoint is also not bundled with the code. We therefore construct the missing parts ourselves while keeping the attack implementation as close as possible to the authors' release.

For this baseline, the victim model is trained for **one epoch** on CelebA using the `Smiling` attribute. The reconstruction target is taken from the **CelebA test split**, meaning that the victim has not seen this image during training. The target is also selected directly by dataset index without first checking whether its class agrees with the class assumption in the authors' reconstruction code.

This setup intentionally gives us a direct starting point rather than modifying the experiment in advance to favor reconstruction. If the reconstruction fails, the result gives us a baseline from which specific possible causes can be investigated in subsequent experiments.

## 2. Methodology / Pipeline / Implementation

The experiment begins by obtaining the authors' GGSS-R implementation. It clones the authors' public repository and extracts its `code.zip` bundle. The expected runner, diffusion implementation, measurement operator, victim model, and configuration files are checked before proceeding.

The victim model follows the authors' `MLP_1` architecture. It is a three-layer fully connected network operating on \(256\times256\) RGB images:

$$
3\times256\times256
\rightarrow 512
\rightarrow 128
\rightarrow 2.
$$

The model contains approximately 100 million parameters. It is trained on the CelebA training split for one epoch using AdamW with batch size 64, learning rate \(10^{-4}\), and weight decay \(10^{-5}\). The resulting checkpoint is then evaluated on the held-out CelebA test split to verify that the victim model is functional before it is used for reconstruction.

Images follow the preprocessing expected by the authors' implementation:

```text
Resize(256 × 256)
-- ToTensor()
-- Normalize(mean=0.5, std=0.5)
```

The reconstruction target is `CelebA test[0]`. Its original image and metadata are saved as experiment artifacts, while the authors' runner performs the corresponding resize and normalization when loading the target.

For the diffusion prior, the experiment uses `ffhq_10m.pt`, the FFHQ diffusion checkpoint expected by the authors' code. Because the checkpoint is not included in the repository, it is downloaded from a public mirror and verified using its expected SHA-256 hash before use.

The attack itself is then executed through the authors' `sample_condition_same_inputs.py` runner using:

* GSS conditioning;
* the authors' 1000-step DDIM configuration;
* guidance rate

  $$
  m_r=0.20;
  $$
* batch size 1 for gradient acquisition;
* no additional gradient perturbation, i.e. the runner uses the target gradient directly.

Only two small changes are made to the authors' source code. The reconstruction progress directory is created before the sampler writes to it, and the diffusion trajectory is saved sparsely rather than at every timestep. These changes affect output handling only; the GSS calculation, gradient guidance, diffusion equations, and reconstruction logic are left unchanged.

The reconstruction is evaluated using both the saved noisy diffusion states \(x_t\) and the predicted clean-image states \(x_0\). MSE, PSNR, and LPIPS are calculated against the target image at the saved timesteps. This allows the experiment to evaluate not only the final reconstruction but also how image similarity changes throughout the reverse-diffusion trajectory.

## 3. Results / Outcomes / Observations / Interpretations

The baseline victim model trained successfully and reached **86.97% training accuracy and 86.84% held-out test accuracy** after one epoch. The target was successfully loaded and processed, and the full 1000-step GGSS-R reconstruction completed successfully using the authors' code path.

The reconstruction itself, however, did not recover the target image.

The quantitative trajectory shows a particularly important pattern. The final predicted \(x_0\) at timestep 0 has:

* MSE: **0.006668**
* PSNR: **21.76 dB**
* LPIPS: **0.4560**

These values are substantially better than the later three-epoch experiment, but they do not by themselves establish successful reconstruction. More importantly, the saved trajectory shows that the reconstruction quality is not steadily improving in all image-space metrics as the diffusion process progresses. The best predicted-\(x_0\) image under all three reported metrics occurs at the final saved timestep \(t=0\), while earlier states show increasing deviation from the target as they move backward through the diffusion trajectory.

The visual reconstruction also does not provide a clean recovery of the target. No natural facial features can be discerned visually, even from the last reconstructed image. The attack therefore does not reproduce the strong reconstruction behavior reported by the authors under this independently constructed victim setup.

At this stage, however, the failure has several possible explanations. The victim was trained for only one epoch, the target was a held-out image rather than a training member, and the target was selected without explicitly checking the class assumption embedded in the authors' released reconstruction code. In addition, the diffusion checkpoint used here was obtained from a public mirror rather than being supplied directly with the GGSS-R repository.

The next experiment therefore changes the two most obvious victim/target conditions: the victim is trained for longer, and the target is explicitly selected as a class-0 member of the victim's training set. The purpose is to test whether a more strongly fitted victim and a training-member target produce a more image-specific leaked gradient and improve reconstruction.
