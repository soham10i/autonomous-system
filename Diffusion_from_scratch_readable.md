# Diffusion Models — A Clean, Readable Guide

This is a cleaned, readable version of your original notes: a structured, concise guide to diffusion models with the main math, intuition, and references. It keeps the important derivations and workflow explanations while removing noisy HTML and duplicated footnotes.

---

## Contents

- Introduction and core idea
- Forward (noising) process — math and intuition
- Reverse (denoising) process — learning to generate
- Training objective (ELBO) and the simplified noise-prediction loss
- Connections: score matching, Langevin dynamics
- Continuous view: SDEs, VP/VE, probability flow ODE
- Major variants and practical advances (DDPM, DDIM, Latent Diffusion, Classifier-Free Guidance, Consistency Models, Flow Matching, EDM)
- Network architecture and conditioning
- Worked example: a single pixel's journey (training → sampling)
- Key references

---

## 1. Introduction — core intuition

Diffusion models are generative models that learn to create data by reversing a gradual noising process. The forward process progressively corrupts data with Gaussian noise. A neural network is trained to reverse that process (denoise) at every noise level. Starting from pure noise and applying the learned reverse steps yields new samples.

Main advantages: stable training, strong sample quality, and flexibility (can be conditioned on text, images, etc.).


## 2. Forward diffusion (discrete-time)

Let $x_0\sim q(x_0)$ be a data sample (e.g., an image). The forward (noising) Markov chain for $t=1\ldots T$ is

$$
q(x_t\mid x_{t-1}) = \mathcal{N}\bigl(x_t;\,\sqrt{1-\beta_t}\,x_{t-1},\;\beta_t\,I\bigr),
$$

where $\{\beta_t\}$ is a user-chosen variance schedule ($0<\beta_t<1$).


Define

$$
\alpha_t \triangleq 1-\beta_t,\qquad \bar{\alpha}_t \triangleq \prod_{s=1}^t \alpha_s.
$$

p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t),\qquad p(x_T)=\mathcal{N}(0,I).
# Diffusion Models: A Complete Mathematical and Theoretical Guide

This document reproduces the full content of your original notes but with cleaned formatting, removed HTML/hidden markup, and clearer LaTeX math blocks so equations are easy to read.

![Complete workflow of diffusion models showing forward noising process (training) and reverse denoising process (generation)](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/6a71f050b273a5a476a2bc98c23b58b7/4fcf0ab3-36aa-4708-9867-2d7e110bbf17/a99daac4.png)


Diffusion models have emerged as one of the most powerful classes of generative models in deep learning, achieving state-of-the-art results in image synthesis, audio generation, and numerous other domains. This comprehensive guide provides a rigorous mathematical foundation, detailed derivations, and an exploration of different types of diffusion models with references to key research papers.

## Introduction and Core Intuition

Diffusion models are generative models inspired by non-equilibrium thermodynamics that learn to generate data by reversing a gradual noising process. The fundamental idea is elegantly simple yet mathematically profound: we systematically destroy structure in data through progressive noise addition (forward process), then train a neural network to reverse this corruption (reverse process).

The breakthrough insight is that if we can learn to denoise data at arbitrary noise levels, we can start from pure random noise and iteratively remove noise to generate new, realistic samples. This approach differs fundamentally from GANs (which use adversarial training) and VAEs (which compress data into latent representations), offering more stable training and higher-quality generation.

## Mathematical Foundations

### The Forward Diffusion Process

The forward process gradually adds Gaussian noise to data over $T$ timesteps, forming a Markov chain that progressively destroys structure until the data becomes indistinguishable from pure noise.

Markov chain formulation:

$$
q(\mathbf{x}_t\mid\mathbf{x}_{t-1}) = \mathcal{N}\bigl(\mathbf{x}_t;\sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\ \beta_t\,\mathbf{I}\bigr),
$$

where $\beta_t\in(0,1)$ is a variance schedule controlling noise magnitude at timestep $t$.

Define

$$
\alpha_t \triangleq 1-\beta_t,\qquad \bar{\alpha}_t \triangleq \prod_{s=1}^t \alpha_s.
$$

Using Gaussian composition, the marginal of $\mathbf{x}_t$ given $\mathbf{x}_0$ has the closed form

$$
q(\mathbf{x}_t\mid\mathbf{x}_0) = \mathcal{N}\bigl(\mathbf{x}_t;\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\ (1-\bar{\alpha}_t)\,\mathbf{I}\bigr).
$$

Thus we can sample $\mathbf{x}_t$ directly from $\mathbf{x}_0$ by the reparameterization

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

The reparameterization trick separates deterministic scaling from stochastic noise and is convenient for training.

Mathematical derivation (sketch): start with

$$
\mathbf{x}_1 = \sqrt{\alpha_1}\mathbf{x}_0 + \sqrt{1-\alpha_1}\,\boldsymbol{\epsilon}_1,
$$

then iterate the Markov step for $\mathbf{x}_2$ and combine independent Gaussian noise terms by summing variances; by induction this yields the closed-form marginal above.

### The Reverse Diffusion Process

The reverse process learns to denoise, starting from pure noise $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ and progressively removing noise to generate data. The reverse conditional is modeled as

$$
p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t) = \mathcal{N}\bigl(\mathbf{x}_{t-1};\,\boldsymbol{\mu}_\theta(\mathbf{x}_t,t),\,\boldsymbol{\Sigma}_\theta(\mathbf{x}_t,t)\bigr).
$$

The generative model is

$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T)\prod_{t=1}^T p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t),\qquad p(\mathbf{x}_T)=\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

When conditioned on $\mathbf{x}_0$, the true reverse posterior is tractable:

$$
q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0) = \mathcal{N}\bigl(\mathbf{x}_{t-1};\,\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0),\,\tilde{\beta}_t\,\mathbf{I}\bigr),
$$

with closed-form expressions (derived by Bayes and completing the square):

$$
	ilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,\mathbf{x}_0 + \frac{\sqrt{\alpha_t}\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,\mathbf{x}_t,
$$

and

$$
	ilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t.
$$

These forms motivate parameterizations of $\mu_\theta$ and choices of $\Sigma_\theta$ during training.

### Training Objective: Variational Lower Bound

We maximize a lower bound on the data log-likelihood (ELBO). Starting from

$$
\log p_\theta(\mathbf{x}_0) = \log \int p_\theta(\mathbf{x}_{0:T})\,d\mathbf{x}_{1:T}
$$

and using the forward process $q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)$ as the approximate posterior, Jensen's inequality yields

$$
\log p_\theta(\mathbf{x}_0) \geq \mathbb{E}_{q}\Big[\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}\Big] = -\mathcal{L}_{VLB}.
$$

One convenient expansion is

$$
\mathcal{L}_{VLB} = \mathbb{E}_q\Big[ -\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1) + \sum_{t=2}^T D_{\mathrm{KL}}\bigl(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\,\|\,p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\bigr) + D_{\mathrm{KL}}\bigl(q(\mathbf{x}_T\mid\mathbf{x}_0)\,\|\,p(\mathbf{x}_T)\bigr)\Big].
$$

For Gaussian conditionals these KL terms have closed forms, so the ELBO can be evaluated or bounded tractably.

### Simplified Training Objective

Ho et al. (DDPM) found that optimizing a simplified noise-prediction objective works very well:

$$
\mathcal{L}_{simple} = \mathbb{E}_{t\sim\mathcal{U}(1,\dots,T),\,\mathbf{x}_0,\,\boldsymbol{\epsilon}}\big[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\|^2\big],
$$

where

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

The network predicts the noise $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)$; from that prediction we can compute a denoised mean and sample using—for example—the update

$$
\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Bigl(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\Bigr) + \sigma_t\,\mathbf{z},\qquad \mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

Setting $\sigma_t=0$ in certain schemes yields deterministic DDIM-like trajectories.

### Connection to Score-Based Models

Diffusion models are tightly connected to score-based generative modeling. The score of the marginal $q(\mathbf{x}_t)$ satisfies

$$
\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\,\mathbb{E}[\boldsymbol{\epsilon}\mid\mathbf{x}_t].
$$

If the model predicts $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\approx\mathbb{E}[\boldsymbol{\epsilon}\mid\mathbf{x}_t]$, then a score estimate is

$$
s_\theta(\mathbf{x}_t,t) \triangleq -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t) \approx \nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t).
$$

Score-based models use such estimates in Langevin dynamics updates or to simulate reverse-time SDEs. A single Langevin update takes the form

$$
\mathbf{x} \leftarrow \mathbf{x} + \eta\,\nabla_{\mathbf{x}}\log p(\mathbf{x}) + \sqrt{2\eta}\,\boldsymbol{\xi},\qquad \boldsymbol{\xi}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

### Stochastic Differential Equation (SDE) Formulation

Song et al. unified diffusion models under a continuous-time SDE framework. The forward SDE is

$$
d\mathbf{x} = f(\mathbf{x},t)\,dt + g(t)\,d\mathbf{w}_t,
$$

where $\mathbf{w}_t$ is a standard Wiener process. The corresponding reverse-time SDE is

$$
d\mathbf{x} = \bigl(f(\mathbf{x},t) - g(t)^2\,\nabla_{\mathbf{x}}\log p_t(\mathbf{x})\bigr)\,dt + g(t)\,d\bar{\mathbf{w}}_t,
$$

where $\bar{\mathbf{w}}_t$ is a reverse-time Wiener process and $p_t(\mathbf{x})$ denotes the marginal at time $t$. This formulation connects discrete DDPM-style models to continuous score-based models.

Two common parameterizations are:

- Variance-Preserving (VP) SDE (analogous to DDPM schedules):

  $$
  d\mathbf{x} = -\tfrac{1}{2}\beta(t)\,\mathbf{x}\,dt + \sqrt{\beta(t)}\,d\mathbf{w}_t.
  $$

- Variance-Exploding (VE) SDE:

  $$
  d\mathbf{x} = \sqrt{\tfrac{d[\sigma^2(t)]}{dt}}\,d\mathbf{w}_t,
  $$

and there exists a deterministic probability-flow ODE that shares the same marginals as the SDE and allows deterministic sampling / exact likelihood computation.

## Different Types of Diffusion Models

### Denoising Diffusion Probabilistic Models (DDPM)

DDPM (Ho et al., 2020) is the foundational discrete-time formulation. Key properties:

- Markovian reverse process $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$.
- Usually trained with $T\approx1000$ timesteps for high-quality samples.
- Stochastic sampling with injected noise at each step.
- Simplified noise-prediction objective as above.

### Denoising Diffusion Implicit Models (DDIM)

DDIM (Song et al.) generalizes sampling from the same trained model to non-Markovian, possibly deterministic trajectories. The DDIM update predicts $\mathbf{x}_0$ and can be written as

$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t) + \sigma_t\,\boldsymbol{\epsilon}_t,
$$

where

$$
\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)}{\sqrt{\bar{\alpha}_t}}.
$$

Setting $\sigma_t=0$ yields deterministic generation and allows much faster sampling (fewer steps) with minimal quality loss in many cases.

### Score-Based Generative Models

Score-based models directly estimate score functions across noise scales and simulate reverse SDEs or use Langevin dynamics for sampling. Denoising score matching is the common training objective.

### Latent Diffusion Models (Stable Diffusion)

Latent diffusion (Rombach et al.) moves the diffusion process into a learned latent space (VAE). The pipeline is:

1. Encode image $\mathbf{x}$ with VAE encoder $\mathcal{E}$ to latent $\mathbf{z}=\mathcal{E}(\mathbf{x})$.
2. Run diffusion (noise & denoise) in the latent space on $\mathbf{z}$.
3. Decode with VAE decoder $\mathcal{D}$ to reconstruct $\mathbf{x}=\mathcal{D}(\mathbf{z})$.

Operating in latent space drastically reduces dimensionality and computational cost, enabling high-resolution synthesis (e.g., Stable Diffusion).

### Classifier-Free Guidance

Classifier-free guidance trains a conditional model $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t,\mathbf{c})$ and randomly drops the conditioning $\mathbf{c}$ during training so the model learns both conditional and unconditional behavior. At sampling time we use:

$$
	ilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t,t,\mathbf{c}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t,\varnothing) + w\bigl(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t,\mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t,\varnothing)\bigr),
$$

where $w$ is the guidance scale (larger $w$ increases fidelity to condition but can reduce diversity).

### Consistency Models

Consistency models (Song et al.) aim for one-step or few-step generation by learning a mapping that is self-consistent across timesteps. They can be trained via distillation from multi-step samplers to achieve extremely fast sampling with good quality.

### Flow Matching

Flow matching learns a time-dependent vector field $\mathbf{v}_\theta(\mathbf{x},t)$ that defines an ODE $d\mathbf{x}/dt=\mathbf{v}_\theta(\mathbf{x},t)$ and trains the field to match conditional transport paths between noise and data. This yields efficient ODE-based samplers and strong likelihoods.

### Elucidating Diffusion Models (EDM)

EDM (Karras et al.) analyzed design choices (schedules, preconditioning, network conditioning) and proposed practical improvements that significantly boost sample quality and efficiency.

## Network Architecture: U-Net with Attention

Most high-quality models use U-Net backbones with timestep embeddings and attention (self-attention and cross-attention for conditioning). Timestep embeddings (sinusoidal or learned) are injected via FiLM / adaptive normalization or added to feature maps.

Self-attention captures long-range dependencies; cross-attention injects conditioning (text, image embeddings) into image features.

## Workflow Example: A Pixel's Journey Through Diffusion

Training phase (learning to denoise):

1. Start with clean image $\mathbf{x}_0$; pick random timestep $t$.
2. Produce noisy sample

   $$
   \mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}.
   $$

3. Feed $\mathbf{x}_t$ and $t$ into the U-Net; predict $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)$.
4. Compute loss $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2$ and backpropagate.

Generation phase (noise → image):

1. Start from $\mathbf{x}_T\sim\mathcal{N}(0,I)$.
2. For $t=T,\dots,1$ compute $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)$ and apply the denoising update to obtain $\mathbf{x}_{t-1}$.
3. After $t=1\to0$ obtain $\mathbf{x}_0$, the generated image.

Intuitively, a single pixel's value evolves from random noise to coherent color as neighboring pixels and learned priors guide denoising.

## Conclusion

Diffusion models synthesize ideas from stochastic processes, score matching, and deep learning to produce powerful, flexible generative models. From DDPMs to latent diffusion and consistency models, the field has matured rapidly with both theoretical and practical advances.

## References (select)

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. arXiv:2006.11239. https://arxiv.org/abs/2006.11239
- Song, Y., & Ermon, S. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. https://arxiv.org/abs/2011.13456
- Song, Y., et al. (2021). Denoising Diffusion Implicit Models (DDIM). https://arxiv.org/abs/2010.02502
- Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion. https://arxiv.org/abs/2112.10752
- Ho, J., & Salimans — Classifier-Free Guidance (2022). https://arxiv.org/abs/2207.12598
- Karras, T., et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. https://arxiv.org/abs/2206.00364
- Lipman et al. — Flow matching (2022+). https://arxiv.org/abs/2210.02747
- Consistency models (2023). https://arxiv.org/abs/2303.01469
*** End Patch
## 10. Key references (select)

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. arXiv:2006.11239. https://arxiv.org/abs/2006.11239
- Song, Y., & Ermon, S. (2019/2021). Score-based generative modeling / generative modeling via SDEs. https://arxiv.org/abs/2011.13456 and related ICLR work.
- Song, Y., et al. (2021). Denoising Diffusion Implicit Models (DDIM). https://arxiv.org/abs/2010.02502
- Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion (Stable Diffusion). https://arxiv.org/abs/2112.10752
- Ho, J., & Salimans — Classifier-Free Guidance (2022). https://arxiv.org/abs/2207.12598
- Karras, T., et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. https://arxiv.org/abs/2206.00364
- Song, Y., et al. (2021). Score-Based Generative Modeling Through SDEs. https://arxiv.org/abs/2011.13456
- Lipman et al. — Flow matching papers (2022+). https://arxiv.org/abs/2210.02747
- Consistency models (2023). https://arxiv.org/abs/2303.01469

(Your original file had many helpful links; these are the core papers to start with.)

---

## Next steps / options

- I saved this cleaned version at:

  `D:\OTH Amberg\Study\sem-2\Autonomous Robots\Maze01\Diffusion_from_scratch_readable.md`

- If you want, I can:
  - Replace your original downloaded file with this cleaned version (need confirmation).
  - Produce a shortened summary (one-page cheat sheet).
  - Convert to PDF or Jupyter Notebook with inline runnable snippets.
  - Expand any section into a deeper derivation (e.g., full ELBO derivation step-by-step).

Tell me which of these you'd like next.
