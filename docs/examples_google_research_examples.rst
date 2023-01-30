Google Research examples
========================

A collection of research by Google Research made with Flax. 

Variational Diffusion Models
****************************

- `Code on GitHub <https://github.com/google-research/vdm/tree/main>`_, `Colab notebooks <https://github.com/google-research/vdm/tree/main/colab>`_ 
- Research paper: `Variational Diffusion Models <https://arxiv.org/abs/2107.00630>`_ (Kingma et al., 2021)
- Introduces *"a family of diffusion-based generative models that obtain state-of-the-art likelihoods on standard image density estimation benchmarks."*

Amos Optimizer and JEstimator
*****************************

- `Code on GitHub <https://github.com/google-research/jestimator>`_
- Research paper: `Amos: An Adam-style Optimizer with Adaptive Weight Decay towards Model-Oriented Scale <https://arxiv.org/abs/2210.11693>`_ (Tian and Parikh, 2022)
- Presents *"Amos, an optimizer compatible with the optax library, and JEstimator, a light-weight library with a tf.Estimator-like interface to manage T5X-compatible checkpoints for machine learning programs in JAX."* *"When used for pre-training BERT variants and T5, Amos consistently converges faster than the state-of-the-art settings of AdamW, achieving better validation loss within <=70% training steps and time, while requiring <=51% memory for slot variables."*

Vision Transformer (ViT) and MLP-Mixer Architectures
****************************************************

- `Code on GitHub <https://github.com/google-research/vision_transformer>`_, `Colab notebooks <https://github.com/google-research/vision_transformer#colab>`_
- Research papers:

  - `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_ (Dosovitskiy et al., 2020)
  - `MLP-Mixer: An all-MLP Architecture for Vision <https://arxiv.org/abs/2105.01601>`_ (Tolstikhin et al., 2021)
  - `How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers <https://arxiv.org/abs/2106.10270>`_ (Steiner et al., 2021)
  - `When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations <https://arxiv.org/abs/2106.01548>`_ (X. Chen et al., 2021)
  - `LiT: Zero-Shot Transfer with Locked-image Text Tuning <https://arxiv.org/abs/2111.07991>`_ (X. Zhai et al., 2021)
  - `Surrogate Gap Minimization Improves Sharpness-Aware Training <https://arxiv.org/abs/2203.08065>`_ (J. Zhuang et al., 2022)

Scaling Vision with Sparse Mixture of Experts (MoE)
***************************************************

- `Code on GitHub <https://github.com/google-research/vmoe>`_
- Research paper: `Scaling Vision with Sparse Mixture of Experts <https://arxiv.org/abs/2106.05974>`_ (Riquelme et al., 2021)
- *"We present a Vision MoE (V-MoE), a sparse version of the Vision Transformer, that is scalable and competitive with the largest dense networks... we demonstrate the potential of V-MoE to scale vision models, and train a 15B parameter model that attains 90.35% on ImageNet."*

Fast Attention (FAVOR+)
***********************

- `Code on GitHub <https://github.com/google-research/google-research/tree/master/performer/fast_attention>`_

  - Implements Performer's Fast Attention (FAVOR+) Module.

- Research paper: `Rethinking Attention with Performers <https://arxiv.org/abs/2009.14794>`_ (Choromanski et al., 2020)
- Introduces *"Performers, Transformer architectures which can estimate regular (softmax) full-rank-attention Transformers with provable accuracy, but using only linear (as opposed to quadratic) space and time complexity, without relying on any priors such as sparsity or low-rankness. To approximate softmax attention-kernels, Performers use a novel Fast Attention Via positive Orthogonal Random features approach (FAVOR+), which may be of independent interest for scalable kernel methods. FAVOR+ can be also used to efficiently model kernelizable attention mechanisms beyond softmax."*