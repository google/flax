# The Flax mission

**The Flax Team** has the goal to incubate, support and advise on a curated collection of JAX libraries. We aim for this curated collection to become **the most researcher-friendly deep learning ecosystem**.

---

## Background: JAX: Numerical computation that is fast, differentiable, and massively-parallel

[JAX](https://jax.readthedocs.io/en/latest/) offers a numerical computation and autodiff library that differs from existing solutions in a few ways. Combined together, these key features lead to a new space in which we can explore ML library design and push the envelope of forward-thinking research.

1. **Compiled functions, which generate efficient GPU or TPU programs thanks to the advanced compiler techniques built into XLA.** This leads to close-to-ideal performance for arbitrary computations. Benchmarks on some of [Hugging Face's JAX models](https://huggingface.co/transformers/#supported-frameworks) show that this can lead to substantial [performance gains](https://github.com/huggingface/transformers/tree/master/examples/flax/text-classification#runtime-evaluation) in practice compared to non-compiled alternatives on both GPU and TPU.

2. **Flexible parallelization primitives.** [JAX](https://jax.readthedocs.io/en/latest/index.html) can automatically vectorize an arbitrary computation; run across many accelerators and machines in parallel; and allows you to write a distributed computation as if it were running on just one device. JAX’s flexible parallelization primitives make it the perfect fit for training very large models.

3. **Composable program transformations allow adding advanced functionality onto existing models, without performance loss.** Examples include: automatic vectorization, advanced higher-order autodiff, and parallelization. We believe that there is a symbiotic relationship between tools and research directions -- when entirely new tools appear, new research directions follow. For example, differential privacy requires efficient per-example gradients, which works out of the box in JAX by composing the vectorization and gradient transformations.

4. **Functional programming offers a fundamentally different way to design libraries.** Pure functions, as opposed to stateful objects, don't require deep coupling that depends on internals. Instead, you can compose any two functions by simply passing the output of one as the input to another. Pure functions have become the norm for most interfaces in libraries built on top of JAX. For instance, [Optax](https://github.com/deepmind/optax) and [Jraph](https://github.com/deepmind/jraph) are libraries based on composable functions, allowing them to be fully independent of which JAX neural network library you use. Additionally, a functional design promotes explicitness and lacks internal state or “spooky action at a distance”. We believe this approach leads to fewer bugs and more robust and interpretable codebases.

---

## Flax: The missing parts for neural networks with JAX

JAX is an excellent basis for current research in many fields in numerical computing--particularly with neural networks. To directly serve various classes of researchers who use neural networks with JAX, it is necessary to have an ecosystem of libraries, carefully designed to build around JAX's philosophy.

At the Flax team, we aim to ensure the JAX ecosystem has the necessary tools and guidance to facilitate and maximize this new space of research directions in the form of libraries (such as [Scenic](https://github.com/google-research/scenic/)), integrated examples for both educational purposes, as well as [ready-to-fork reference implementations](https://github.com/google/flax/tree/main/examples), [guides](https://flax.readthedocs.io/en/latest/guides/index.html), support channels, and best practices.

Flax aims to support researchers' needs, while exposing the full power of JAX, rather than encapsulating it away. We think that this is the best path towards ground-breaking research that builds on JAX's unique capabilities. Some recent examples of innovative research projects on top of Flax include [PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html) (a 540-billion parameter model for text generation), [Imagen](https://imagen.research.google/) (text-to-image diffusion models), [T5X](https://github.com/google-research/t5x) for very large scale language model, [Brax](https://github.com/google/brax) for differentiable on-device rigid body simulations, and [NetKet](https://www.netket.org/) for many-body quantum systems.

It is challenging to design a library that is ergonomic for researchers, while building on a purely functional library like JAX, which may be confusing to some. But we see this challenge as **an opportunity**. How do you design a smooth user experience where the complexity is progressively disclosed? How do you help users avoid "silent footguns", such as accidentally mutating batch norm stats during prediction? How do you make sure errors guide users to the right solution, even when there may be multiple correct ones? How do you design examples and tutorials that are simple enough to be educational, while being good enough to act as forkable starter codebases? These are the types of questions that inform our design process.

As members of the Flax team, we spend a majority of our time on **community engagement**. To make sure what we build is delightful for our current and future users, we hold design discussions in the open. This helps the community share ideas and engage with Flax core developers directly, while we learn from each others' complementary experiences and integrate the feedback. We [actively engage](https://github.com/google/flax/pull/1011) with users via GitHub [issues](https://github.com/google/flax/issues), [discussions](https://github.com/google/flax/discussions) and [pull requests](https://github.com/google/flax/pulls). We share [the Flax philosophy and design notes](https://flax.readthedocs.io/en/latest/philosophy.html), and [recommended best practices](https://flax.readthedocs.io/en/latest/guides/index.html) with the community. We improve our APIs based on feedback, while supporting our users through any transition—take a look at how we did it for our recent ["Linen" rewrite](https://github.com/google/flax/tree/main/flax/linen), as an example.

---

## Growing the Flax Team

JAX and Flax have been growing rapidly, both in open source and within Google, and we're looking to grow the Flax team. With thoughtful technical and organizational structures in place, we hope to facilitate a 10x growth of the Flax user base. Due to the flexibility of interop through pure functions, we can imagine a broader Flax effort, with a flourishing ecosystem of libraries, to serve both an increased number of users, as well as **entirely new forms of research**.

We see all of our work stemming from our foundational goal: **to serve our users**. We are constantly developing a deeper understanding of our users' needs and challenges, and we do whatever it takes to supply our users with the best experience. We discuss challenges and possible solutions with our users, especially when the right choice isn't apparent -- and these discussions lead us to continuous improvements. These improvements take many forms, such as improved documentation, error messages, or new solutions to common problems.

Does this mission excite you? If so, take a look at [our job posting](https://careers.google.com/jobs/results/116638751486026438-software-engineer-jax-and-flax-google-research/) [despite what it says, we'd consider candidates in Amsterdam, Zurich or the San Francisco Bay Area]. And even if you're not sure if the job posted fits your background exactly, or if now isn't the best time, please do reach out to us at [join-flax@google.com](mailto:join-flax@google.com)  — we'd love to hear from you and get to know you better.

With great hopes for the future,

Marc, Jonathan, George, Bertrand, Avital, Anselm, Andreas,

on behalf of The Flax Team
