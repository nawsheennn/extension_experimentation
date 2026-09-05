# Victim Model Comparison for Reconstruction

## 1. Motivation / Objective

The previous `gradient_identifiability_analyses` experiment narrowed the reconstruction failure to the victim model itself.

Earlier diagnostics had already shown that the diffusion model can generate natural face images, that the GSS execution path remains on the natural image manifold, and that the leaked-gradient objective can be optimized successfully. Direct gradient inversion then showed that a candidate could become extremely close to the leaked gradient without recovering the target image. The identifiability analysis traced this behavior to weak image-specific gradients reaching the input-facing layer of the authors' single-epoch `MLP_1`.

This experiment asks whether reconstruction quality changes when the **victim architecture** and **training state** are changed.

Two model states are compared:

* **epoch 0**: the randomly initialized model before any training;
* **epoch 1**: the same model after one epoch of CelebA `Smiling` classification training.

Epoch 0 is useful because it separates gradient identifiability from learned classifier confidence. An untrained model has not yet learned the classification task, so its loss can remain substantial. However, a large loss alone does not guarantee useful image gradients: that error signal must still survive backpropagation through the network and reach layers that retain information about the input.

Four custom architectures are tested:

* `mlp_small`: a **25.2M-parameter MLP**. MLP victims are reported as relatively reconstructable in GGSS-R, and the first fully connected layer has a direct pixel-to-weight relationship with the input.
* `lenet_sigmoid`: a severely under-parameterized **34K-parameter CNN**, used to test whether limited capacity affects the amount of image-specific information available in the gradient.
* `cnn_gi`: a moderate **4.3M-parameter CNN**, representing a middle ground in capacity.
* `cnn_overparam`: a large **35.1M-parameter CNN**, used to examine whether substantially increasing convolutional model capacity improves or worsens gradient identifiability.

These architectural choices are hypotheses to test rather than expected outcomes. The reconstruction results determine which model states actually expose enough image-specific gradient information for inversion.

The experiment uses **direct gradient inversion** rather than GGSS-R. The target gradients are completely unperturbed, allowing the effect of victim architecture and training state to be isolated without introducing diffusion sampling or gradient perturbation.

This does not make diffusion guidance unnecessary. GGSS-R is designed for the harder setting where leaked gradients may be perturbed. Direct inversion can reproduce noisy features of that perturbation because it only tries to satisfy the observed gradient. A diffusion prior instead constrains the candidate toward the natural image manifold, which can make reconstruction more robust to noisy gradients. The purpose here is simply to first establish which clean victim gradients are identifiable.

## 2. Methodology / Pipeline / Implementation

The experiment is implemented across two notebooks for execution convenience. `victim_model_comparison_training.ipynb` creates and trains the victim checkpoints, while `victim_model_comparison_reconstruction.ipynb` performs the gradient diagnostics and reconstruction attacks. Together they form one experiment.

The same class-0 CelebA training-member target used in the preceding reconstruction diagnostics is retained. Images are resized to $256\times256$, converted to tensors, and normalized to \([-1,1]\).

### Victim architectures

`mlp_small` contains one 128-unit hidden layer:

$$
3\times256\times256
\rightarrow128
\rightarrow2.
$$

Its first fully connected layer accounts for almost all of its 25,166,210 parameters.

`lenet_sigmoid` contains four sigmoid convolutional layers followed directly by a two-class fully connected layer. It contains only 34,022 parameters.

`cnn_gi` contains five sigmoid convolutional layers followed by a 128-unit fully connected hidden layer and a two-class output layer, for 4,334,114 parameters.

`cnn_overparam` follows a similar structure with substantially wider convolutional layers and a 512-unit hidden layer, giving 35,106,946 parameters.

For each architecture, the randomly initialized weights are first saved as the epoch-0 checkpoint. The same model is then trained for one epoch using cross-entropy loss and Adam with learning rate \(0.001\).

After one epoch, the training outcomes are:

| Model           | Parameters | Training loss | Training accuracy |
| --------------- | ---------: | ------------: | ----------------: |
| `mlp_small`     | 25,166,210 |        0.5445 |            73.38% |
| `lenet_sigmoid` |     34,022 |        0.7099 |            51.34% |
| `cnn_gi`        |  4,334,114 |        0.7046 |            51.53% |
| `cnn_overparam` | 35,106,946 |        0.7186 |            50.67% |

Only `mlp_small` learns the classification task substantially within the single epoch. The three CNNs remain close to chance-level accuracy.

### Target-gradient construction

For every architecture and epoch, the target image is passed through the victim and its classification loss is differentiated with respect to every model parameter:

$$
g_{\text{target}} = \nabla_\theta \mathcal{L}\left(f_\theta(x_{\text{target}}), y_{\text{target}}\right)
$$

The gradient norm of every parameter tensor is also recorded. This is important because the total gradient magnitude alone does not show whether useful signal reaches the layers that interact most directly with the image.

### Direct gradient inversion

Each reconstruction begins from a random image. The candidate is passed through the same frozen victim and differentiated using the known target label:

$$
g(x) = \nabla_\theta \mathcal{L}\left(f_\theta(x), y_{\text{target}}\right)
$$

The candidate image is then optimized to minimize the mean squared difference between the complete candidate and target gradients:

$$
\mathcal{L}_{\text{inv}} = \text{MSE}\left(g(x), g_{\text{target}}\right)
$$

The gradient computation remains differentiable so that this objective can be backpropagated through the victim and into the candidate image.

Each architecture × epoch combination is tested using three independent random seeds, giving:

$$
4\text{ architectures} \times 2\text{ epochs} \times 3\text{ seeds} = 24
$$

reconstruction runs.

Every run uses:

* 1,500 Adam optimization steps;
* learning rate \(0.1\);
* the complete, unperturbed victim gradient;
* candidate clipping to \([-1,1]\).

Gradient-space progress is tracked using relative gradient distance and cosine similarity. 
Image reconstruction is evaluated independently using pixel MSE, PSNR, and LPIPS. 
The full optimization trajectories are retained so that changes in gradient space can be compared directly with changes in image space.

## 3. Results / Outcomes / Observations / Interpretations

### Reconstruction results

The full architecture × epoch × seed sweep produces one clear successful condition: `mlp_small` at epoch 0.

Across all three seeds, the initially random image gradually converges toward the target in both gradient space and image space.

| Model state              |    Mean MSE |    Mean PSNR | Mean LPIPS |
| ------------------------ | ----------: | -----------: | ---------: |
| `mlp_small`, epoch 0     | **0.00113** | **29.72 dB** | **0.1189** |
| `mlp_small`, epoch 1     |     0.16397 |      7.88 dB |     1.2020 |
| `lenet_sigmoid`, epoch 0 |     0.15109 |      8.21 dB |     1.2704 |
| `lenet_sigmoid`, epoch 1 |     0.12478 |      9.04 dB |     1.2781 |
| `cnn_gi`, epoch 0        |     0.15194 |      8.18 dB |     1.2714 |
| `cnn_gi`, epoch 1        |     0.15194 |      8.18 dB |     1.2714 |
| `cnn_overparam`, epoch 0 |     0.15194 |      8.18 dB |     1.2714 |
| `cnn_overparam`, epoch 1 |     0.15194 |      8.18 dB |     1.2714 |

The three successful `mlp_small` epoch-0 runs finish with MSEs of 0.00154, 0.00116, and 0.00068. 
Their PSNR values reach 28.12, 29.37, and 31.67 dB, while LPIPS falls to 0.164, 0.127, and 0.066.

The trajectories show the same pattern. 
Relative gradient distance falls from approximately 1.0 to 0.083–0.137, cosine similarity rises from approximately 0.04 to 0.991–0.997, and image MSE falls from approximately 0.151 to near zero.

The successful reconstruction is therefore consistent across all three random initializations.

The same architecture behaves very differently at epoch 1. All three runs fail to reconstruct the target. 
Two remain essentially at their starting image-space error, while one becomes worse. Mean MSE rises to 0.164, PSNR falls below 8 dB, and LPIPS remains above 1.2.

The three CNN architectures also fail to produce useful reconstructions. For `cnn_gi` and `cnn_overparam`, both epoch 0 and epoch 1 trajectories are completely flat: pixel MSE remains around 0.152 for all 1,500 steps.

`lenet_sigmoid` at epoch 0 behaves similarly, with MSE changing only from approximately 0.152 to 0.151. Its epoch-1 runs show some numerical optimization, reducing MSE to approximately 0.125, but perceptual quality does not improve: LPIPS remains around 1.28. This is therefore not a successful image reconstruction.

### Observation 1: The epoch-0 MLP contains a strong input-facing gradient.

The layer norms explain why the untrained `mlp_small` behaves differently.

At epoch 0, its target loss is 0.6189 and its total gradient norm is 21.37. Almost all of that useful signal reaches the first layer:

$$
\|\nabla W_{\text{fc1}}\| = 21.05
$$

For a fully connected first layer,

$$
z_1=W_1x+b_1
$$

the weight gradient for a single input has the form

$$
\frac{\partial \mathcal{L}}{\partial W_1} = \delta_1 x^\top
$$

The image therefore appears directly within the first-layer gradient whenever the backpropagated error \(\delta_1\) remains sufficiently strong.

That condition is present in the untrained MLP. Its gradient objective provides a strong image-specific direction, and the candidate moves steadily toward the target.

### Observation 2: One epoch sharply weakens the MLP gradient.

After one epoch of training, the `mlp_small` target loss falls from 0.6189 to 0.2372.

Its total target-gradient norm simultaneously falls:

$$
21.37 \rightarrow 4.62
$$

while the first-layer weight-gradient norm drops even more clearly:

$$
21.05 \rightarrow 3.89
$$

The reconstruction quality collapses at the same time.

For this model, the result matches the behavior identified in the previous `gradient_identifiability_analyses`: as the victim becomes more confident on the target, the classification loss and backpropagated error become smaller, leaving much less input-specific gradient signal for inversion.

The epoch-0 versus epoch-1 MLP comparison therefore shows that training state alone can substantially change reconstruction vulnerability even when the architecture is unchanged.
The authors' paper did not address training state. 
They established `Reconstruction Vulnerability` as a metric of how susceptible models are to gradient inversion / data reconstruction attacks,
and they showed a comparison of model architectures with their RV values, with the 3-layered MLP being the easiest to reconstruct from.
However, they did not address how reconstruction vulnerability changes depending on the training state of the same model architecture.
Their reported RV values are therefore not a static metric of model architectural vulnerability.

### Observation 3: The CNNs already suffer severe gradient attenuation at epoch 0.

The CNN results show that an untrained model is not automatically reconstructable.

Their epoch-0 target losses are all substantial:

* `lenet_sigmoid`: 0.6504;
* `cnn_gi`: 0.6476;
* `cnn_overparam`: 0.9045.

Despite this, the gradient reaching the early feature layers is extremely weak.

For `lenet_sigmoid`, the total epoch-0 gradient norm is actually the largest among the four models:

$$
38.02.
$$

However, almost all of this norm is concentrated in the final classifier:

$$
\|\nabla W_{\text{fc}}\| = 38.01.
$$

The first convolutional layer has a weight-gradient norm of only:

$$
0.00150.
$$

The same pattern is stronger in the deeper CNNs.

For `cnn_gi` at epoch 0:

$$
\|\nabla g\|_{\text{total}} = 9.65,
$$

but

$$
\|\nabla W_{\text{conv1}}\|
\approx 1.0\times10^{-5}.
$$

For `cnn_overparam`:

$$
\|\nabla g\|_{\text{total}} = 18.39,
$$

while

$$
\|\nabla W_{\text{conv1}}\|
\approx 1.5\times10^{-5}.
$$

Most of their gradient energy instead appears in the later fully connected layers.

The problem is therefore not simply whether the loss or total gradient is large. The error signal must survive the complete backpropagation route and remain informative in layers connected to the image.

All three CNNs contain repeated sigmoid nonlinearities between the output and the early feature layers. Their measured layer norms progressively shrink toward the input, which is consistent with severe gradient attenuation through that deeper backpropagation path. By the time the gradient reaches the layers responsible for extracting information directly from the image, very little usable signal remains.

This explains why reconstruction can fail even for an untrained model with a substantial classification loss.

### Observation 4: Training further collapses the CNN gradient paths.

The epoch-1 layer norms show an even stronger version of the same problem.

For `lenet_sigmoid`, the total gradient norm falls from 38.02 to 2.73. Its final fully connected layer still carries most of that gradient, while the feature-layer gradients remain very small.

The effect is most extreme for `cnn_gi`.

At epoch 1, every gradient from the first convolutional layer through `fc1` is exactly zero in the recorded target gradient. Only the final classifier remains nonzero:

$$
\|\nabla W_{\text{fc2}}\| = 5.84.
$$

`cnn_overparam` shows the same structure. Every feature-layer and `fc1` gradient is zero, while only `fc2` remains active:

$$
\|\nabla W_{\text{fc2}}\| = 11.23.
$$

For these two trained CNNs, there is therefore no gradient path from the reconstruction objective back through the victim to the image-producing layers. The candidate cannot receive an image-specific optimization signal.


### What this experiment establishes

This experiment resolves the central question raised by `gradient_identifiability_analyses`.

Clean victim gradients **can** contain enough information for high-quality image reconstruction. The three successful `mlp_small` epoch-0 runs demonstrate this directly.

However, gradient identifiability depends strongly on **both the victim architecture and its state**.
