# GGSS-R Training-Member Reconstruction with a More-Trained Victim

## 1. Motivation / Objective

This experiment follows the baseline GGSS-R reproduction and tests whether the victim model and target selection used there were responsible for the poor reconstruction.

In the baseline experiment, the victim was trained for only one epoch and the reconstruction target was taken from the CelebA test split. The target therefore had not been seen by the victim during training. The target was also selected without checking its `Smiling` label, even though the authors' released reconstruction operator contains a class-0 assumption.

This experiment changes both conditions.

First, the victim is trained for **three epochs instead of one**. The motivation is that a more strongly fitted victim may memorize its training examples more effectively. If the leaked gradient becomes more image-specific as the victim fits the training data more closely, reconstruction could become easier.

Second, the target is explicitly selected from the **training split** and required to have `Smiling == 0`. This makes the target a genuine member of the victim's training data while also satisfying the class convention assumed by the authors' released reconstruction code.

The experiment therefore tests the hypothesis that:

> a more strongly trained victim combined with a training-member target may produce a more image-specific leaked gradient and improve GGSS-R reconstruction.

The GGSS-R attack itself is not redesigned. The goal is to determine whether changing the victim-training and target-membership conditions is enough to recover the target.

## 2. Methodology / Pipeline / Implementation

The experiment uses the same authors' `MLP_1` victim architecture and GGSS-R reconstruction pipeline as the baseline.

The victim is trained on the complete CelebA training split using the same preprocessing and optimization settings as before:

* batch size: 64;
* AdamW learning rate: \(10^{-4}\);
* weight decay: \(10^{-5}\);
* training duration: 3 epochs.

A checkpoint is saved after every completed epoch, with the third-epoch checkpoint used as the canonical attack checkpoint. The victim contains 100,729,730 parameters.

The target is selected deterministically as the first CelebA training example with `Smiling == 0`. This image is therefore both:

1. a member of the data used to train the victim, and
2. consistent with the class-0 assumption in the authors' ReconstructionOperator.

The target is saved once as an experiment artifact and reused throughout the reconstruction.

The same FFHQ diffusion checkpoint used in the baseline experiment is used here. It is downloaded and SHA-256 verified before being placed at the path expected by the authors' code.

The reconstruction again uses the authors' GSS implementation with:

$$
m_r=0.20,
$$

1000 DDIM diffusion steps, and no additional gradient noise. The notebook makes only three compatibility/output changes:

* represents the hard-coded class-0 target as a `LongTensor` class index;
* creates the reconstruction progress directory before it is used;
* saves sparse $x_t$ and predicted $x_0$ snapshots every 100 steps and at $t=0$.

The underlying GSS calculation and GGSS-R sampling procedure remain unchanged.

After reconstruction, the saved trajectory is evaluated using MSE, PSNR, and LPIPS against the target. Both the noisy $x_t$ states and the predicted clean $x_0$ states are evaluated so that the reconstruction can be inspected numerically and visually across the full diffusion process.

## 3. Results / Outcomes / Observations / Interpretations

The three-epoch victim trained successfully. Its training loss decreased from **0.3066 after epoch 1 to 0.2384 after epoch 3**, and the final held-out test accuracy was **89.97%**. The target was confirmed to be a class-0 training member, and the trained victim also classified it as class 0.

Thus, the two changes motivated by the baseline experiment were successfully implemented: the victim was more strongly trained, and the target was appropriately selected.

They did **not** improve the reconstruction.

The full 1000-step GGSS-R run completed, but the reconstruction trajectory remained poor. At the final predicted $x_0$, the image-space metrics were:

* MSE: **0.172187**
* PSNR: **7.64 dB**
* LPIPS: **0.718507**

The visual reconstruction is even more revealing: the final outputs do not recover meaningful features of the target face and instead become dominated by patches of color and distorted structures.

Therefore, the hypothesis tested here is not supported. Making the victim substantially more trained and choosing a genuine training-member target did not rescue GGSS-R reconstruction; in this run, the result was substantially worse than the first baseline.

This changes the direction of the investigation. The failure is no longer well explained simply by the victim being insufficiently trained or the target being outside the training set. We now need to examine the reconstruction pipeline itself more carefully.

The next experiment, `ggssr_unperturbed_reproduction_diagnostic`, therefore moves away from changing the victim and instead performs a more controlled reproduction of the GGSS-R setup. In particular, it uses the FFHQ checkpoint referenced by the DPS implementation and adds a series of pre-reconstruction diagnostics to test the target gradient, the gradient of unrelated images, second-order differentiability, the gradient behavior of an actual diffusion sample, and finally the authors' original ReconstructionOperator. These checks are intended to determine whether the reconstruction failure can be localized to a specific component of the pipeline.
