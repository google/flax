Repositories that use Flax
==========================

The following code bases use Flax and provide training frameworks and a wealth
of examples. In many cases, you can also find pre-trained weights:


🤗 Hugging Face
***************

`🤗 Hugging Face <https://huggingface.co/flax-community>`__ is a
very popular library for building, training, and deploying state of the art
machine learning models.
These models can be applied on text, images, and audio. After organizing the
`JAX/Flax community week <https://github.com/huggingface/transformers/blob/master/examples/research_projects/jax-projects/README.md>`__,
they have now over 5,000
`Flax/JAX models <https://huggingface.co/models?library=jax&sort=downloads>`__ in
their repository.

🥑 DALLE Mini
*************

`🥑 DALLE Mini <https://huggingface.co/dalle-mini>`__ is a Transformer-based
text-to-image model implemented in JAX/Flax that follows the ideas from the
original `DALLE <https://openai.com/blog/dall-e/>`__ paper by OpenAI.

Scenic
******

`Scenic <https://github.com/google-research/scenic>`__ is a codebase/library
for computer vision research and beyond. Scenic's main focus is around
attention-based models. Scenic has been successfully used to develop
classification, segmentation, and detection models for multiple modalities
including images, video, audio, and multimodal combinations of them.

Big Vision
**********

`Big Vision <https://github.com/google-research/big_vision/>`__ is a codebase
designed for training large-scale vision models using Cloud TPU VMs or GPU
machines. It is based on Jax/Flax libraries, and uses tf.data and TensorFlow
Datasets for scalable and reproducible input pipelines. This is the original
codebase of ViT, MLP-Mixer, LiT, UViM, and many more models.

T5X
***

`T5X <https://github.com/google-research/t5x>`__ is a modular, composable,
research-friendly framework for high-performance, configurable, self-service
training, evaluation, and inference of sequence models (starting with
language) at many scales.