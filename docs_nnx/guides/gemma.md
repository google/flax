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

# Getting Started with Gemma Sampling using NNX: A Step-by-Step Guide

You will find in this colab a detailed tutorial explaining how to use NNX to load a Gemma checkpoint and sample from it.

+++

## Installation

```{code-cell} ipython3
! pip install --no-deps -U flax
! pip install jaxtyping kagglehub penzai
```

## Downloading the checkpoint

"To use Gemma's checkpoints, you'll need a Kaggle account and API key. Here's how to get them:

1. Visit https://www.kaggle.com/ and create an account.
2. Go to your account settings, then the 'API' section.
3. Click 'Create new token' to download your key.

Then run the cell below.

```{code-cell} ipython3
import kagglehub
kagglehub.login()
```

If everything went well, you should see:
```
Kaggle credentials set.
Kaggle credentials successfully validated.
```

Now select and download the checkpoint you want to try. Note that you will need an A100 runtime for the 7b models.

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

Flax examples are not exposed as packages so you need to use the workaround in the next cells to import from NNX's Gemma example.

```{code-cell} ipython3
! git clone https://github.com/google/flax.git flax_examples
```

```{code-cell} ipython3
import sys

sys.path.append("./flax_examples/flax/nnx/examples/gemma")
import params as params_lib
import sampler as sampler_lib
import transformer as transformer_lib
sys.path.pop();
```

## Start Generating with Your Model

Load and prepare your LLM's checkpoint for use with Flax.

```{code-cell} ipython3
:cellView: form

# Load parameters
params = params_lib.load_and_format_params(ckpt_path)
```

Load your tokenizer, which we'll construct using the [SentencePiece](https://github.com/google/sentencepiece) library.

```{code-cell} ipython3
:cellView: form

vocab = spm.SentencePieceProcessor()
vocab.Load(vocab_path)
```

Use the `transformer_lib.TransformerConfig.from_params` function to automatically load the correct configuration from a checkpoint. Note that the vocabulary size is smaller than the number of input embeddings due to unused tokens in this release.

```{code-cell} ipython3
transformer = transformer_lib.Transformer.from_params(params)
nnx.display(transformer)
```

Finally, build a sampler on top of your model and your tokenizer.

```{code-cell} ipython3
:cellView: form

# Create a sampler with the right param shapes.
sampler = sampler_lib.Sampler(
    transformer=transformer,
    vocab=vocab,
    params=params['transformer'],
)
```

You're ready to start sampling ! This sampler uses just-in-time compilation, so changing the input shape triggers recompilation, which can slow things down. For the fastest and most efficient results, keep your batch size consistent.

```{code-cell} ipython3
:cellView: form

input_batch = [
    "\n# Python program for implementation of Bubble Sort\n\ndef bubbleSort(arr):",
    "What are the planets of the solar system?",
  ]

out_data = sampler(
    input_strings=input_batch,
    total_generation_steps=300,  # number of steps performed when generating
  )

for input_string, out_string in zip(input_batch, out_data.text):
  print(f"Prompt:\n{input_string}\nOutput:\n{out_string}")
  print()
  print(10*'#')
```

You should get an implementation of bubble sort and a description of the solar system.
