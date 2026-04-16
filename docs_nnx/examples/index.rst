Examples
########

NNX documentation contains two kinds of examples: example notebooks, which gradually introduce new concepts
for applying NNX to different application areas, and example projects, which are more realistic representations
of how nontrivial models should be implemented.


Example Notebooks
=================

Example notebooks guide you through applying Flax models to a variety of different domains.

.. toctree::
   :maxdepth: 1

   ./gemma
   ./digits_diffusion_model


Example Projects
================

Example projects are hosted on the GitHub Flax repository in the `examples <https://github.com/google/flax/tree/main/examples>`__
directory.

Each example is designed to be **self-contained and easily forkable**, while
reproducing relevant results in different areas of machine learning.

Transformers
********************

- :octicon:`mark-github;0.9em` `Gemma <https://github.com/google/flax/tree/main/examples/gemma/>`__ :
  A family of open-weights Large Language Model (LLM) by Google DeepMind, based on Gemini research and technology.
  Gemma models training and evaluation script on the One Billion Word Benchmark (LM1B).

Toy examples
********************

`NNX toy examples <https://github.com/google/flax/tree/main/examples/nnx_toy_examples/>`__
directory contains a few smaller, standalone toy examples for simple training scenarios.
