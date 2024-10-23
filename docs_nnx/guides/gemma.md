---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

Copyright 2024 The Flax Authors.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

---

+++

# Get started with Gemma sampling/inference using Flax NNX

In this tutorial, you will learn step-by-step how to use Flax NNX to load the [Gemma](https://ai.google.dev/gemma) open model files and use them to perform sampling/inference for generating text. You will use the [Flax NNX `gemma` code](https://github.com/google/flax/tree/main/examples/gemma) for model parameter configuration and inference that was written with Flax and JAX.

> Gemma is a family of lightweight, state-of-the-art open models based on Google DeepMind’s [Gemini](https://deepmind.google/technologies/gemini/#introduction). Read more about [Gemma](https://blog.google/technology/developers/gemma-open-models/) and [Gemma 2](https://blog.google/technology/developers/google-gemma-2/).

You are recommended to use [Google Colab](https://colab.research.google.com/) with TPU v2-8 acceleration to run the code.

Let’s get started.

+++

## Installation

Install the necessary dependencies, including `kagglehub`.

```{code-cell} ipython3
! pip install --no-deps -U flax
! pip install jaxtyping kagglehub penzai
```

## Download the model

To use Gemma model, you'll need a [Kaggle](https://www.kaggle.com/models/google/gemma/) account and API key:

1. To create an account, visit [Kaggle](https://www.kaggle.com/) and click on 'Register'.
2. If/once you have an account, you need to sign in, go to your ['Settings'](https://www.kaggle.com/settings), and under 'API' click on 'Create New Token' to generate and download your Kaggle API key.
3. OPTIONAL: In [Google Colab](https://colab.research.google.com/), under 'Secrets' add your Kaggle username and API key, storing the username as `KAGGLE_USERNAME` and the key as `KAGGLE_KEY`. If you are using a [Kaggle Notebook](https://www.kaggle.com/code) for free TPU or other hardware acceleration, it has a key storage feature under 'Add-ons' > 'Secrets', along with instructions for accessing stored keys.

Then run the cell below.

```{code-cell} ipython3
import kagglehub
kagglehub.login()
```

If everything went well, it should say `Kaggle credentials set. Kaggle credentials successfully validated.`.

**Note:** In Google Colab, you can instead authenticate into Kaggle using the code below after following the optional step 3 from above.

```
import os
from google.colab import userdata # `userdata` is a Colab API.

os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
``` 

Now, load the Gemma model you want to try. The code in the next cell utilizes [`kagglehub.model_download`](https://github.com/Kaggle/kagglehub/blob/8efe3e99477aa4f41885840de6903e61a49df4aa/src/kagglehub/models.py#L16) to download model files.

**Note:** For larger models, such as `gemma 7b` and `gemma 7b-it` (instruct), you may require a hardware accelerator with plenty of memory, such as the NVIDIA A100.

```{code-cell} ipython3
VARIANT = '2b-it' # @param ['2b', '2b-it', '7b', '7b-it'] {type:"string"}
weights_dir = kagglehub.model_download(f'google/gemma/Flax/{VARIANT}')
ckpt_path = f'{weights_dir}/{VARIANT}'
vocab_path = f'{weights_dir}/tokenizer.model'
```

## Python imports

```{code-cell} ipython3
from flax import nnx
import sentencepiece as spm
```

To interact with the Gemma model, you will use the Flax NNX `gemma` code from [`google/flax` examples on GitHub](https://github.com/google/flax/tree/main/examples/gemma). Since it is not exposed as a package, you need to use the following workaround to import from the Flax NNX `gemma` example.

```{code-cell} ipython3
! git clone https://github.com/google/flax.git flax_examples
```

Import wrapper modules and functions:

```{code-cell} ipython3
import sys

sys.path.append("./flax_examples/examples/gemma")
import params as params_lib
import sampler as sampler_lib
import transformer as transformer_lib
sys.path.pop();
```

## Load and prepare the Gemma model

First, load the Gemma model parameters for use with Flax.

```{code-cell} ipython3
:cellView: form

params = params_lib.load_and_format_params(ckpt_path)
```

Next, load the tokenizer file constructed using the [SentencePiece](https://github.com/google/sentencepiece) library.

```{code-cell} ipython3
:cellView: form

vocab = spm.SentencePieceProcessor()
vocab.Load(vocab_path)
```

Then, use the Flax NNX [`gemma.transformer.TransformerConfig.from_params`](https://github.com/google/flax/blob/3f3c03b23d4fd3d85d1c5d4d97381a8a2c48b475/examples/gemma/transformer.py#L193) function to automatically load the correct configuration from a checkpoint.

**Note:** The vocabulary size is smaller than the number of input embeddings due to unused tokens in this release.

```{code-cell} ipython3
transformer = transformer_lib.Transformer.from_params(params)
nnx.display(transformer)
```

## Perform sampling/inference

Build a Flax NNX [`gemma.Sampler`](https://github.com/google/flax/blob/main/examples/gemma/sampler.py) on top of your model and tokenizer with the right parameter shapes.

```{code-cell} ipython3
:cellView: form

sampler = sampler_lib.Sampler(
    transformer=transformer,
    vocab=vocab,
    params=params['transformer'],
)
```

You're ready to start sampling!

**Note:** This Flax NNX [`gemma.Sampler`](https://github.com/google/flax/blob/main/examples/gemma/sampler.py) uses JAX’s [just-in-time (JIT) compilation](https://jax.readthedocs.io/en/latest/jit-compilation.html), so changing the input shape triggers recompilation, which can slow things down. For the fastest and most efficient results, keep your batch size consistent.

Write a prompt in `input_batch` and perform inference. Feel free to tweak `total_generation_steps` (the number of steps performed when generating a response).

```{code-cell} ipython3
:cellView: form

input_batch = [
    "\n# Python program for implementation of Bubble Sort\n\ndef bubbleSort(arr):",
    "What are the planets of the solar system?",
  ]

out_data = sampler(
    input_strings=input_batch,
    total_generation_steps=300,  # The number of steps performed when generating a response.
  )

for input_string, out_string in zip(input_batch, out_data.text):
  print(f"Prompt:\n{input_string}\nOutput:\n{out_string}")
  print()
  print(10*'#')
```

You should get a Python implementation of the bubble sort algorithm and a description of planets in the solar system.
