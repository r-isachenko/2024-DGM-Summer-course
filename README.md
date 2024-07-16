# Deep Generative Models summer course, 2024

## Description
The course is devoted to modern generative models (mostly in the application to computer vision).

We will study the following types of generative models:
- autoregressive models,
- latent variable models,
- normalization flow models,
- adversarial models,
- diffusion models.

Special attention is paid to the properties of various classes of generative models, their interrelationships, theoretical prerequisites and methods of quality assessment.

The aim of the course is to introduce the student to widely used advanced methods of deep learning.

The course is accompanied by practical tasks that allow you to understand the principles of the considered models.

## Contact the author to join the course or for any other questions :)

- **telegram:** [@roman_isachenko](https://t.me/roman_isachenko)
- **e-mail:** roman.isachenko@phystech.edu

## Materials

| # | Date | Description | Slides |
|---|---|---|---|
| 1 | June, 25 | <b>Lecture 1:</b> Logistics. Generative models overview and motivation. Problem statement. Divergence minimization framework. Autoregressive models. | [slides](lectures/lecture1/Lecture1.pdf) |
|  |  | <b>Seminar 1:</b> Introduction. Maximum likelihood estimation. Histograms. Bayes theorem. | [slides](seminars/seminar1/seminar1.ipynb) |
| 2 | June, 27 | <b>Lecture 2:</b> Autoregressive models (PixelCNN). Normalizing Flow (NF) intuition and definition. Forward and reverse KL divergence for NF. Linear NF. Gaussian autoregressive NF. | [slides](lectures/lecture2/Lecture2.pdf) |
|  |  | <b>Seminar 2:</b> PixelCNN. | [slides](seminars/seminar2/seminar2.ipynb) <a href="https://colab.research.google.com/github/r-isachenko/2024-DGM-Summer-course/blob/main/seminars/seminar2/seminar2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>|
| 3 | July, 2 | <b>Lecture 3:</b> Linear NF. Gaussian autoregressive NF. Coupling layer (RealNVP).  | [slides](lectures/lecture3/Lecture3.pdf) |
|  |  | <b>Seminar 3:</b> Planar and Radial Flows. Forward vs Reverse KL. | [slides](seminars/seminar3/seminar3.ipynb)  <a href="https://colab.research.google.com/github/r-isachenko/2024-DGM-Summer-course/blob/main/seminars/seminar3/seminar3.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> PlanarFlow notebook: <a href="https://colab.research.google.com/github/r-isachenko/2024-DGM-Summer-course/blob/main/seminars/seminar3/planar_flow.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>|
| 4 | July, 4 | <b>Lecture 4:</b> Latent Variable Models (LVM). Variational lower bound (ELBO). Variational EM-algorithm. Amortized inference. | [slides](lectures/lecture4/Lecture4.pdf) |
|  |  | <b>Seminar 4:</b> RealNVP. | [slides](seminars/seminar4/real_nvp_notes.ipynb) |
| 5 | July, 9 | <b>Lecture 5:</b> ELBO gradients, reparametrization trick. Variational Autoencoder (VAE). NF as VAE model. ELBO surgery and optimal VAE prior. | [slides](lectures/lecture5/Lecture5.pdf) |
|  |  | <b>Seminar 5:</b> Gaussian Mixture Model (GMM). GMM and MLE. ELBO and EM-algorithm. GMM via EM-algorithm. Variational EM algorithm for GMM. | [slides](seminars/seminar5/seminar5.ipynb) |
| 6 | July, 11 | <b>Lecture 6:</b> NF-based VAE prior. Discrete VAE latent representations. Vector quantization, straight-through gradient estimation (VQ-VAE). | [slides](lectures/lecture6/Lecture6.pdf) |
|  |  | <b>Seminar 6:</b>  VAE: Implementation hints. Vanilla 2D VAE coding. VAE on Binarized MNIST visualization. | [slides](seminars/seminar6/seminar6.ipynb) |
<!---
| 7 | July, 16 | <b>Lecture 7:</b> Likelihood-free learning. GAN optimality theorem. Wasserstein distance. | [slides](lectures/lecture7/Lecture7.pdf) |
|  |  | <b>Seminar 7:</b> Posterior collapse. Beta VAE on MNIST. | [slides](seminars/seminar7/seminar7.ipynb) |
| 8 | July, 18 | <b>Lecture 8:</b> Wasserstein GAN (WGAN). WGAN with gradient penalty (WGAN-GP). f-divergence minimization. | [slides](lectures/lecture8/Lecture8.pdf) |
|  |  | <b>Seminar 8:</b> KL vs JS divergences. Vanilla GAN in 1D coding. Mode collapse and vanishing gradients. Non-saturating GAN. | [slides](seminars/seminar8/seminar8.ipynb) |
| 9 | July, 23 | <b>Lecture 9:</b> GAN evaluation. FID, MMD, Precision-Recall, truncation trick. Langevin dynamic. Score matching. | [slides](lectures/lecture9/Lecture9.pdf) |
|  |  | <b>Seminar 9:</b> WGAN and WGAN-GP on 1D data. | [slides](seminars/seminar9/seminar9.ipynb) |
| 10 | July, 25 | <b>Lecture 10:</b> Denoising score matching. Noise Conditioned Score Network (NCSN). Gaussian diffusion process: forward + reverse. | [slides](lectures/lecture10/Lecture10.pdf) |
|  |  | <b>Seminar 10:</b> StyleGAN. | [slides](seminars/seminar10/StyleGAN.ipynb) |
| 11 | July, 30 | <b>Lecture 11:</b> Gaussian diffusion model as VAE, derivation of ELBO. Reparametrization of gaussian diffusion model. | [slides](lectures/lecture11/Lecture11.pdf) |
|  |  | <b>Seminar 11:</b> Noise Conditioned Score Network (NCSN). Gaussian diffusion model as VAE. | [slides](seminars/seminar11/seminar11.ipynb) |
| 12 | August, 2 | <b>Lecture 12:</b> Denoising diffusion probabilistic model (DDPM): overview. Denoising diffusion as score-based generative model. Model guidance: classifier guidance, classfier-free guidance. | [slides](lectures/lecture12/Lecture12.pdf) |
|  |  | <b>Seminar 12:</b> Denoising diffusion probabilistic model (DDPM). Denoising Diffusion Implicit Models (DDIM). | [slides](seminars/seminar11/seminar11.ipynb) |
| 13 | August, 6 | <b>Lecture 13:</b> Continuous-in-time NF and neural ODE. Kolmogorov-Fokker-Planck equation for NF log-likelihood. FFJORD and Hutchinson's trace estimator. Adjoint method for continuous-in-time NF. SDE basics. Kolmogorov-Fokker-Planck equation. Probability flow ODF. Reverse SDE. Variance Preserving and Variance Exploding SDEs. | [slides](lectures/lecture13/Lecture13.pdf) |
|  |  | <b>Seminar 13:</b> Guidance. CLIP, GLIDE, DALL-E 2, Imagen, Latent Diffusion Model. | [slides](seminars/seminar13/seminar13.ipynb) |
--->

## Homeworks
| Homework | Date | Deadline | Description | Link |
|---------|------|-------------|--------|-------|
| 1 | July, 3 | July, 17 | <ol><li>Theory (alpha-divergences, curse of dimensionality, NF expressivity).</li><li>ImageGPT on MNIST.</li><li>RealNVP on CIFAR10.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw1.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2024-DGM-Summer-course/blob/main/homeworks/hw1.ipynb) |
| 2 | July, 17 | July, 31 | <ol><li>Theory (IWAE theory, Conjugate functions, FID for Normal distributions).</li><li>ResNetVAE on CIFAR10.</li><li>WGAN/WGAN-GP on CIFAR10.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw2.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2024-DGM-AIMasters-course/blob/main/homeworks/hw2.ipynb) |
<!---
| 3 | July, 31 | August, 14 | <ol><li>Theory (IWAE theory, MI in ELBO surgery, Gumbel-Max trick).</li><li>ResNetVAE on CIFAR10.</li><li>VQ-VAE with PixelCNN prior.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw3.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2024-DGM-AIMasters-course/blob/main/homeworks/hw3.ipynb) |
--->

## Game rules
- 6 homeworks each of 13 points = **78 points**
- oral cozy exam = **26 points**
- maximum points: 78 + 26 = **104 points**
### Final grade: `floor(relu(#points/8 - 2))`

## Prerequisities
- probability theory + statistics
- machine learning + basics of deep learning
- python + basics of one of DL frameworks (pytorch/tensorflow/etc)

## Previous episodes
- [2024, spring, AIMasters](https://github.com/r-isachenko/2024-DGM-AIMasters-course)
- [2023, autumn, MIPT](https://github.com/r-isachenko/2023-DGM-MIPT-course)
- [2022-2023, autumn-spring, MIPT](https://github.com/r-isachenko/2022-2023-DGM-MIPT-course)
- [2022, autumn, AIMasters](https://github.com/r-isachenko/2022-2023-DGM-AIMasters-course)
- [2022, spring, OzonMasters](https://github.com/r-isachenko/2022-DGM-Ozon-course)
- [2021, autumn, MIPT](https://github.com/r-isachenko/2021-DGM-MIPT-course)
- [2021, spring, OzonMasters](https://github.com/r-isachenko/2021-DGM-Ozon-course)
- [2020, autumn, MIPT](https://github.com/r-isachenko/2020-DGM-MIPT-course)

