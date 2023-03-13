########################
Google Research examples
########################

A collection of research by Google Research made with Flax.

Attention
*********

Fast Attention (FAVOR+) and Rethinking Attention with Performers
================================================================

- Code on GitHub:

  - `Performer's Fast Attention (FAVOR+) module <https://github.com/google-research/google-research/tree/master/performer/fast_attention>`__

- Research paper:

  - `Rethinking Attention with Performers <https://arxiv.org/abs/2009.14794>`__ (Choromanski et al., 2020)

    - Introduces *"Performers, Transformer architectures which can estimate regular (softmax) full-rank-attention Transformers with provable accuracy, but using only linear (as opposed to quadratic) space and time complexity, without relying on any priors such as sparsity or low-rankness. To approximate softmax attention-kernels, Performers use a novel Fast Attention Via positive Orthogonal Random features approach (FAVOR+), which may be of independent interest for scalable kernel methods. FAVOR+ can be also used to efficiently model kernelizable attention mechanisms beyond softmax."*

Self-attention Does Not Need O(n^2) Memory
==========================================

- `Code on GitHub <https://github.com/google-research/google-research/tree/master/memory_efficient_attention>`__
- `Colab notebook <https://github.com/google-research/google-research/blob/master/memory_efficient_attention/memory_efficient_attention.ipynb>`__

- Research paper:

  - `Self-attention Does Not Need O(n^2) Memory <https://arxiv.org/abs/2112.05682>`__ (Rabe and Staats, 2021)

    - *"We present a very simple algorithm for attention that requires O(1) memory with respect to sequence length and an extension to self-attention that requires O(log n) memory. This is in contrast with the frequently stated belief that self-attention requires O(n^2) memory. While the time complexity is still O(n^2), device memory rather than compute capability is often the limiting factor on modern accelerators. Thus, reducing the memory requirements of attention allows processing of longer sequences than might otherwise be feasible..."*

Computer vision
***************

Colorization Transformer (ColTran)
==================================

- `Code on GitHub <https://github.com/google-research/google-research/tree/master/coltran>`__

- Research paper:

  - `Colorization Transformer <https://openreview.net/forum?id=5NA1PinlGFu>`__ (Kumar et al., 2020)

    - *"We presented the Colorization Transformer (ColTran), an architecture that entirely relies on selfattention for image colorization. We introduce conditional transformer layers, a novel building block for conditional, generative models based on self-attention. Our ablations show the superiority of employing this mechanism over a number of different baselines. Finally, we demonstrate that ColTran can generate diverse, high-fidelity colorizations on ImageNet, which are largely indistinguishable from the ground-truth even for human raters."*

Vision Transformer (ViT), MLP-Mixer Architectures *and* Big Vision
==================================================================

- Code on GitHub:

  - `Vision Transformer and MLP-Mixer Architectures <https://github.com/google-research/vision_transformer>`__

  - `Big Vision <https://github.com/google-research/big_vision>`__

    - *"This codebase is designed for training large-scale vision models using Cloud TPU VMs or GPU machines. It is based on Jax/Flax libraries, and uses tf.data and TensorFlow Datasets for scalable and reproducible input pipelines."*

- `Colab notebooks <https://github.com/google-research/vision_transformer#colab>`__:

  - The JAX code of Vision Transformers and MLP Mixers
  - More than 50k Vision Transformer and hybrid checkpoints that were used to generate the data of "How to train your ViT?"

- Research papers:

  - `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`__ (Dosovitskiy et al., 2020)

    - *"In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train."*

  - `MLP-Mixer: An All-MLP Architecture for Vision <https://arxiv.org/abs/2105.01601>`__ (Tolstikhin et al., 2021)

    - *"In this paper we show that while convolutions and attention are both sufficient for good performance, neither of them are necessary. We present MLP-Mixer, an architecture based exclusively on multi-layer perceptrons (MLPs). MLP-Mixer contains two types of layers: one with MLPs applied independently to image patches (i.e. "mixing" the per-location features), and one with MLPs applied across patches (i.e. "mixing" spatial information). When trained on large datasets, or with modern regularization schemes, MLP-Mixer attains competitive scores on image classification benchmarks, with pre-training and inference cost comparable to state-of-the-art models."*

  - `How to Train Your ViT? Data, Augmentation, and Regularization in Vision Transformers <https://arxiv.org/abs/2106.10270>`__ (Steiner et al., 2021)

    - *"Vision Transformers (ViT) have been shown to attain highly competitive performance for a wide range of vision applications, such as image classification, object detection and semantic image segmentation. In comparison to convolutional neural networks, the Vision Transformer's weaker inductive bias is generally found to cause an increased reliance on model regularization or data augmentation ("AugReg" for short) when training on smaller training datasets. We conduct a systematic empirical study in order to better understand the interplay between the amount of training data, AugReg, model size and compute budget."*

  - `When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations <https://arxiv.org/abs/2106.01548>`__ (X. Chen et al., 2021)

    - *"Vision Transformers (ViTs) and MLPs signal further efforts on replacing hand-wired features or inductive biases with general-purpose neural architectures. Existing works empower the models by massive data, such as large-scale pre-training and/or repeated strong data augmentations, and still report optimization-related problems (e.g., sensitivity to initialization and learning rates). Hence, this paper investigates ViTs and MLP-Mixers from the lens of loss geometry, intending to improve the models' data efficiency at training and generalization at inference."*

  - `LiT: Zero-Shot Transfer with Locked-image Text Tuning <https://arxiv.org/abs/2111.07991>`__ (X. Zhai et al., 2021)

    - *"This paper presents contrastive-tuning, a simple method employing contrastive training to align image and text models while still taking advantage of their pre-training. In our empirical study we find that locked pre-trained image models with unlocked text models work best. We call this instance of contrastive-tuning "Locked-image Tuning" (LiT), which just teaches a text model to read out good representations from a pre-trained image model for new tasks. A LiT model gains the capability of zero-shot transfer to new vision tasks, such as image classification or retrieval. The proposed LiT is widely applicable; it works reliably with multiple pre-training methods (supervised and unsupervised) and across diverse architectures (ResNet, Vision Transformers and MLP-Mixer) using three different image-text datasets."*

Scaling Vision with Sparse Mixture of Experts (MoE)
===================================================

- `Code on GitHub <https://github.com/google-research/vmoe>`__
- Research paper:

  - `Scaling Vision with Sparse Mixture of Experts <https://arxiv.org/abs/2106.05974>`__ (Riquelme et al., 2021)

    - *"Sparsely-gated Mixture of Experts networks (MoEs) have demonstrated excellent scalability in Natural Language Processing. In Computer Vision, however, almost all performant networks are "dense", that is, every input is processed by every parameter. We present a Vision MoE (V-MoE), a sparse version of the Vision Transformer, that is scalable and competitive with the largest dense networks... we demonstrate the potential of V-MoE to scale vision models, and train a 15B parameter model that attains 90.35% on ImageNet..."*

Diffusion
*********

Variational Diffusion Models
============================

- `Code on GitHub <https://github.com/google-research/vdm/tree/main>`__
- `Colab notebooks <https://github.com/google-research/vdm/tree/main/colab>`__
- Research paper:

  - `Variational Diffusion Models <https://arxiv.org/abs/2107.00630>`__ (Kingma et al., 2021)

    - *"Diffusion-based generative models have demonstrated a capacity for perceptually impressive synthesis, but can they also be great likelihood-based models? We answer this in the affirmative, and introduce a family of diffusion-based generative models that obtain state-of-the-art likelihoods on standard image density estimation benchmarks. Unlike other diffusion-based models, our method allows for efficient optimization of the noise schedule jointly with the rest of the model. We show that the variational lower bound (VLB) simplifies to a remarkably short expression in terms of the signal-to-noise ratio of the diffused data, thereby improving our theoretical understanding of this model class. Using this insight, we prove an equivalence between several models proposed in the literature. In addition, we show that the continuous-time VLB is invariant to the noise schedule, except for the signal-to-noise ratio at its endpoints. This enables us to learn a noise schedule that minimizes the variance of the resulting VLB estimator, leading to faster optimization..."*

Domain adaptation
*****************

GIFT (Gradual Interpolation of Features toward Target)
======================================================

- `Code on GitHub <https://github.com/google-research/google-research/tree/master/gift>`__
- Research paper:

  - `Gradual Domain Adaptation in the Wild: When Intermediate Distributions are Absent <https://arxiv.org/abs/2106.06080>`__ (Abnar et al., 2021)

    - *"We focus on the problem of domain adaptation when the goal is shifting the model towards the target distribution, rather than learning domain invariant representations. It has been shown that under the following two assumptions: (a) access to samples from intermediate distributions, and (b) samples being annotated with the amount of change from the source distribution, self-training can be successfully applied on gradually shifted samples to adapt the model toward the target distribution. We hypothesize having (a) is enough to enable iterative self-training to slowly adapt the model to the target distribution, by making use of an implicit curriculum. In the case where (a) does not hold, we observe that iterative self-training falls short. We propose GIFT, a method that creates virtual samples from intermediate distributions by interpolating representations of examples from source and target domains..."*

Generalization
**************

Surrogate Gap Minimization Improves Sharpness-Aware Training
============================================================

- `Code on GitHub <https://github.com/google-research/big_vision/tree/main/big_vision/trainers/proj/gsam>`__
- Research paper:

  - `Surrogate Gap Minimization Improves Sharpness-Aware Training <https://arxiv.org/abs/2203.08065>`__ (J. Zhuang et al., 2022)

    - *"The recently proposed Sharpness-Aware Minimization (SAM) improves generalization by minimizing a perturbed loss defined as the maximum loss within a neighborhood in the parameter space. However, we show that both sharp and flat minima can have a low perturbed loss, implying that SAM does not always prefer flat minima. Instead, we define a surrogate gap, a measure equivalent to the dominant eigenvalue of Hessian at a local minimum when the radius of neighborhood (to derive the perturbed loss) is small. The surrogate gap is easy to compute and feasible for direct minimization during training. Based on the above observations, we propose Surrogate Gap Guided Sharpness-Aware Minimization (GSAM), a novel improvement over SAM with negligible computation overhead..."*

Meta learning
*************

``learned_optimization``
=======================

- Code on GitHub: `learned_optimization <https://github.com/google/learned_optimization>`__
- `Colab notebooks <https://github.com/google/learned_optimization#learned_optimization-tutorial-sequence>`__

- Research papers:

  - `Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies <http://proceedings.mlr.press/v139/vicol21a.html>`__ (Vicol et al., 2021)

    - *"We introduce a method called Persistent Evolution Strategies (PES), which divides the computation graph into a series of truncated unrolls, and performs an evolution strategies-based update step after each unroll. PES eliminates bias from these truncations by accumulating correction terms over the entire sequence of unrolls. PES allows for rapid parameter updates, has low memory usage, is unbiased, and has reasonable variance characteristics."*

  - `Gradients Are Not All You Need <https://arxiv.org/abs/2111.05803>`__	(Metz et al., 2021)

    - *"...In this short report, we discuss a common chaos based failure mode which appears in a variety of differentiable circumstances, ranging from recurrent neural networks and numerical physics simulation to training learned optimizers. We trace this failure to the spectrum of the Jacobian of the system under study, and provide criteria for when a practitioner might expect this failure to spoil their differentiation based optimization algorithms."*

Model efficiency
****************

Efficiently Scaling Transformer Inference
=========================================

- Code on GitHub:

  - `T5X <https://github.com/google-research/t5x>`__
  - `AQT: Accurate Quantized Training <http://github.com/google/aqt>`__

- Research paper:

  - `Efficiently Scaling Transformer Inference <https://arxiv.org/abs/2211.05102>`__ (Pope et al., 2022)

    - *"We develop a simple analytical model for inference efficiency to select the best multi-dimensional partitioning techniques optimized for TPU v4 slices based on the application requirements. We combine these with a suite of low-level optimizations to achieve a new Pareto frontier on the latency and model FLOPS utilization (MFU) tradeoffs on 500B+ parameter models that outperforms the FasterTransformer suite of benchmarks. We further show that with appropriate partitioning, the lower memory requirements of multiquery attention (i.e. multiple query heads share single key/value head) enables scaling up to 32× larger context lengths."*

Neural rendering / NeRF
***********************

Generalizable Patch-Based Neural Rendering
==========================================

- `Code on GitHub <https://github.com/google-research/google-research/tree/master/gen_patch_neural_rendering>`__
- Research paper:

  - `Generalizable Patch-Based Neural Rendering <https://arxiv.org/abs/2207.10662>`__ (Suhail et al., 2022)

    - *"...We propose a different paradigm, where no deep features and no NeRF-like volume rendering are needed. Our method is capable of predicting the color of a target ray in a novel scene directly, just from a collection of patches sampled from the scene."*

Voxel-based Radiance Fields in JAX and Flax
===========================================

- `Colab notebook <https://github.com/google-research/google-research/blob/master/trainable_grids/Voxel_based_Radiance_Fields.ipynb>`__ (Velez and Dellaert, 2022)

  - *"In this notebook we show how with JAX/Flax, it is relatively easy to quickly get a voxel-based NeRF variant up and running. Specifically, we will develop a simplified version of DVGO that directly regresses color instead of having a small MLP. It works remarkably well."*

Optimization
************

Amos Optimizer *and* JEstimator
===============================

- Code on GitHub:

  - `Amos and JEstimator <https://github.com/google-research/jestimator>`__

    - *"... implements Amos, an optimizer compatible with the optax library, and JEstimator, a light-weight library with a tf.Estimator-like interface to manage T5X-compatible checkpoints for machine learning programs in JAX, which we use to run experiments in the paper."*

- Research paper:

  - `Amos: An Adam-style Optimizer with Adaptive Weight Decay towards Model-Oriented Scale <https://arxiv.org/abs/2210.11693>`__ (Tian and Parikh, 2022)

    - Presents *"Amos, an optimizer compatible with the optax library, and JEstimator, a light-weight library with a tf.Estimator-like interface to manage T5X-compatible checkpoints for machine learning programs in JAX."* *"When used for pre-training BERT variants and T5, Amos consistently converges faster than the state-of-the-art settings of AdamW, achieving better validation loss within <=70% training steps and time, while requiring <=51% memory for slot variables."*

Quantization
************

Pareto-Optimal Quantized ResNet Is Mostly 4-bit *and* AQT: Accurate Quantized Training
======================================================================================

- Code on GitHub:

  - `AQT: Accurate Quantized Training <http://github.com/google/aqt>`__

- Research paper:

  - `Pareto-Optimal Quantized ResNet Is Mostly 4-bit <https://arxiv.org/abs/2105.03536>`__ (Abdolrashidi et al., 2021)

    - *"In this work, we use ResNet as a case study to systematically investigate the effects of quantization on inference compute cost-quality tradeoff curves. Our results suggest that for each bfloat16 ResNet model, there are quantized models with lower cost and higher accuracy; in other words, the bfloat16 compute cost-quality tradeoff curve is Pareto-dominated by the 4-bit and 8-bit curves, with models primarily quantized to 4-bit yielding the best Pareto curve... The quantization method we used is optimized for practicality: It requires little tuning and is designed with hardware capabilities in mind... As part of this work, we contribute a quantization library written in JAX..."*

Reinforcement learning
**********************

Continuous Control with Action Quantization from Demonstrations (AQuaDem)
=========================================================================

- `Code on GitHub <https://github.com/google-research/google-research/tree/master/aquadem>`__

- Research paper:

  - `Continuous Control with Action Quantization from Demonstrations <https://arxiv.org/abs/2110.10149>`__ (Dadashi et al., 2021)

    - Proposes *"a novel Reinforcement Learning (RL) framework for problems with continuous action spaces: Action Quantization from Demonstrations (AQuaDem). The proposed approach consists in learning a discretization of continuous action spaces from human demonstrations. This discretization returns a set of plausible actions (in light of the demonstrations) for each input state, thus capturing the priors of the demonstrator and their multimodal behavior. By discretizing the action space, any discrete action deep RL technique can be readily applied to the continuous control problem. Experiments show that the proposed approach outperforms state-of-the-art methods such as SAC in the RL setup, and GAIL in the Imitation Learning setup."*

Sequence models / Model parallelism
***********************************

T5X: Scaling Up Models and Data with ``t5x`` and ``seqio``
==========================================================

- `Code on GitHub <https://github.com/google-research/t5x>`__

  - *"T5X is a modular, composable, research-friendly framework for high-performance, configurable, self-service training, evaluation, and inference of sequence models (starting with language) at many scales."*

- Research paper:

  - `T5X: Scaling Up Models and Data with t5x and seqio <https://arxiv.org/abs/2203.17189>`__ (Roberts et al., 2022)

    - *"Recent neural network-based language models have benefited greatly from scaling up the size of training datasets and the number of parameters in the models themselves. Scaling can be complicated due to various factors including the need to distribute computation on supercomputer clusters (e.g., TPUs), prevent bottlenecks when infeeding data, and ensure reproducible results. In this work, we present two software libraries that ease these issues: t5x simplifies the process of building and training large language models at scale while maintaining ease of use, and seqio provides a task-based API for simple creation of fast and reproducible training data and evaluation pipelines. These open-source libraries have been used to train models with hundreds of billions of parameters on datasets with multiple terabytes of training data. Along with the libraries, we release configurations and instructions for T5-like encoder-decoder models as well as GPT-like decoder-only architectures."*

Simulation
**********

Brax - A Differentiable Physics Engine for Large Scale Rigid Body Simulation
============================================================================

- `Code on GitHub <https://github.com/google/brax>`__
- `Colab notebooks <https://github.com/google/brax#quickstart-colab-in-the-cloud>`__
- Research paper:

  - `Brax - A Differentiable Physics Engine for Large Scale Rigid Body Simulation <https://arxiv.org/abs/2106.13281>`__ (Freeman et al., 2021)

    - *"We present Brax, an open source library for rigid body simulation with a focus on performance and parallelism on accelerators, written in JAX. We present results on a suite of tasks inspired by the existing reinforcement learning literature, but remade in our engine. Additionally, we provide reimplementations of PPO, SAC, ES, and direct policy optimization in JAX that compile alongside our environments, allowing the learning algorithm and the environment processing to occur on the same device, and to scale seamlessly on accelerators."*
